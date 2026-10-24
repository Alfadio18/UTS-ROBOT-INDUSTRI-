#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

# ================== INFORMASI GEOMETRI DAN SIMULASI ROBOT ==================
# Ukuran mekanis robot menurut konfigurasi URDF: jari-jari roda 0,03 m dan jarak antar-roda 0,17 m.
PHYS_RADIUS = 0.03
PHYS_SEP = 0.17
# Parameter roda dari plugin penggerak diferensial yang aktif pada lingkungan Gazebo.
PLUGIN_RADIUS = 0.1
PLUGIN_SEP = 0.45

START_POSE = (0.0, 0.0, 0.0)   # posisi awal simulasi (koordinat X, Y, dan orientasi yaw)
WHEEL_SIGN = 1.0               # faktor penyesuaian tanda putaran jika pembacaan roda berlawanan
CMD_GAIN_V = 1.0               # pengali kompensasi kecepatan translasi hasil pengujian gerak
CMD_GAIN_W = 1.0
ODOM_TOPIC = '/odom_gt'        # sumber data posisi dari topik odometri referensi
POSE_TOPIC = '/world_pose'     # alternatif informasi posisi melalui jembatan transformasi
ROBOT_NAME = 'robin'           # identitas model yang dipantau pada dunia simulasi

# ================== RANCANGAN LINTASAN DENGAN WAYPOINT ==================
# Titik navigasi ditetapkan berdasarkan pusat ruang lintasan pada model Gazebo, dalam satuan meter.
def build_waypoints():
    wp = [
        (0.90, 2.35, 0.12),   # titik pendekatan sebelum memasuki lintasan utama
        (2.45, 2.35, 0.10),   # mengikuti ruas awal ke arah kanan
        (2.45, 3.05, 0.10),   # perpindahan jalur menuju sisi atas
        (3.68, 3.05, 0.10),   # penyesuaian posisi ke jalur horizontal selanjutnya
        (3.68, 0.78, 0.10),   # bergerak menuju segmen dengan koordinat Y lebih kecil
        (7.90, 0.78, 0.10),   # melintasi sisi bawah arena menuju bagian kanan
        (7.90, 6.00, 0.10),   # menuju bagian awal tikungan melengkung
    ]
    # Segmen putar balik diaproksimasi oleh rangkaian target yang membentuk kurva semi-elips.
    wp += [
        (7.81, 6.38, 0.10), (7.62, 6.72, 0.10), (7.37, 7.00, 0.10),
        (7.07, 7.20, 0.10), (6.73, 7.32, 0.10), (6.38, 7.36, 0.10),
        (6.02, 7.32, 0.10), (5.68, 7.20, 0.10), (5.38, 6.99, 0.10),
        (5.13, 6.72, 0.10), (4.94, 6.38, 0.10),
    ]
    wp += [
        (4.85, 6.00, 0.15),   # transisi keluar dari belokan melingkar
        (4.85, 4.25, 0.10),   # bergerak menuju segmen dengan koordinat Y lebih kecil
        (3.90, 4.25, 0.10),   # bergerak horizontal ke arah sisi kiri
        (3.90, 6.80, 0.10),   # mengarahkan robot kembali ke bagian atas lintasan
        (3.10, 6.80, 0.15),   # melalui akses sempit untuk memasuki ruang di sisi kiri
        (3.10, 7.55, 0.15),
        (0.275, 7.55, 0.10),  # bergerak horizontal ke arah sisi kiri di atas
        (0.275, 4.85, 0.10),  # posisi akhir perjalanan di ujung tertutup koridor
    ]
    return wp


WAYPOINTS = build_waypoints()

# ================== BATAS GERAK DAN RESPONS PENGENDALI ==================
V_MAX = 0.25        # batas laju linear robot, dalam meter per detik
W_MAX = 1.0         # pembatas laju putaran, dalam radian per detik
KP_ANG = 2.0
ROTATE_ON = 0.60    # sudut kesalahan yang memicu koreksi orientasi di tempat
ROTATE_OFF = 0.10   # toleransi kesalahan sudut sebelum robot meneruskan translasi


def norm(a):
    return math.atan2(math.sin(a), math.cos(a))


class WaypointController:

    def __init__(self, waypoints):
        self.wps = waypoints
        self.idx = 0
        self.rotating = False
        self.done = False

    def step(self, x, y, yaw):
        if self.done:
            return 0.0, 0.0
        gx, gy, tol = self.wps[self.idx]
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        if dist < tol:
            self.idx += 1
            self.rotating = False
            if self.idx >= len(self.wps):
                self.done = True
                return 0.0, 0.0
            gx, gy, tol = self.wps[self.idx]
            dx, dy = gx - x, gy - y
            dist = math.hypot(dx, dy)
        err = norm(math.atan2(dy, dx) - yaw)

        if self.rotating:
            if abs(err) < ROTATE_OFF:
                self.rotating = False
        elif abs(err) > ROTATE_ON:
            self.rotating = True

        if self.rotating:
            w = max(-W_MAX, min(W_MAX, KP_ANG * err))
            if abs(w) < 0.25:
                w = math.copysign(0.25, err)
            return 0.0, w
        v = V_MAX * max(0.0, math.cos(err)) * min(1.0, 0.3 + dist / 0.3)
        w = max(-W_MAX, min(W_MAX, KP_ANG * err))
        return v, w


def physical_to_cmd(v, w):
    wl = (v - w * PHYS_SEP / 2.0) / PHYS_RADIUS     # hasil konversi ke kecepatan sudut roda kiri (rad/s)
    wr = (v + w * PHYS_SEP / 2.0) / PHYS_RADIUS
    cmd_v = PLUGIN_RADIUS * (wl + wr) / 2.0
    cmd_w = PLUGIN_RADIUS * (wr - wl) / PLUGIN_SEP
    return cmd_v * CMD_GAIN_V, cmd_w * CMD_GAIN_W


class TrackRunner(Node):
    def __init__(self):
        super().__init__(
            'track_runner',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)],
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Float64MultiArray, '/wheel/speed', self.wheel_cb, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_cb, 10)
        self.create_subscription(TFMessage, POSE_TOPIC, self.tf_cb, 10)

        self.x, self.y, self.yaw = START_POSE
        self.wl = 0.0
        self.wr = 0.0
        self.last_t = None
        self.ctrl = WaypointController(WAYPOINTS)
        self.last_idx = -1
        self.have_odom = False
        self.abs_source = False
        self.running = False
        self.odom0 = None
        self.t_start = self.get_clock().now().nanoseconds * 1e-9
        self.warned = False
        self.last_warn = -10.0
        self.declare_parameter('require_gt', True)
        self.require_gt = bool(self.get_parameter('require_gt').value)
        self.create_timer(0.05, self.loop)
        self.get_logger().info(f'Navigasi lintasan disiapkan: {len(WAYPOINTS)} titik target. Menanti informasi posisi...')

    @staticmethod
    def quat_to_yaw(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        self.set_true_pose(p.x, p.y, self.quat_to_yaw(msg.pose.pose.orientation),
                           ODOM_TOPIC, absolute=False)

    def tf_cb(self, msg):
        if not msg.transforms:
            return
        for ts in msg.transforms:
            if ts.child_frame_id and ts.child_frame_id.split('::')[-1] == ROBOT_NAME:
                t = ts.transform.translation
                self.set_true_pose(t.x, t.y, self.quat_to_yaw(ts.transform.rotation),
                                   POSE_TOPIC, absolute=True)
                return

        if any(ts.child_frame_id for ts in msg.transforms):
            return
        ts = msg.transforms[0]
        t = ts.transform.translation
        if self.have_odom and math.hypot(t.x - self.x, t.y - self.y) > 0.5:
            return
        self.set_true_pose(t.x, t.y, self.quat_to_yaw(ts.transform.rotation),
                           POSE_TOPIC, absolute=True)

    def set_true_pose(self, px, py, yaw, source, absolute):
        if not self.have_odom:
            self.have_odom = True
            self.abs_source = absolute
            self.odom0 = (px, py, yaw)
            self.get_logger().info(f'Lokalisasi tersambung melalui sumber data {source}')
        if absolute:                 # koordinat global dapat langsung dijadikan acuan navigasi
            self.x, self.y, self.yaw = px, py, yaw
            return
        x0, y0, yaw0 = self.odom0    # transformasi odometri dimulai dari orientasi pembacaan awal
        c0, s0 = math.cos(-yaw0), math.sin(-yaw0)
        rx = c0 * (px - x0) - s0 * (py - y0)
        ry = s0 * (px - x0) + c0 * (py - y0)
        cs, ss = math.cos(START_POSE[2]), math.sin(START_POSE[2])
        self.x = START_POSE[0] + cs * rx - ss * ry
        self.y = START_POSE[1] + ss * rx + cs * ry
        self.yaw = norm(START_POSE[2] + norm(yaw - yaw0))

    def wheel_cb(self, msg):
        if len(msg.data) < 2:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        # perkirakan perpindahan memakai kecepatan roda bila pose referensi belum tersedia
        if self.last_t is not None and not self.have_odom:
            dt = now - self.last_t
            if 0.0 < dt < 0.2:
                v = PHYS_RADIUS * (self.wr + self.wl) / 2.0
                w = PHYS_RADIUS * (self.wr - self.wl) / PHYS_SEP
                self.x += v * math.cos(self.yaw + w * dt / 2.0) * dt
                self.y += v * math.sin(self.yaw + w * dt / 2.0) * dt
                self.yaw = norm(self.yaw + w * dt)
        self.last_t = now
        self.wl = WHEEL_SIGN * msg.data[0]
        self.wr = WHEEL_SIGN * msg.data[1]

    def loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if not self.have_odom:
            if self.require_gt:
                # cegah perintah gerak ketika data lokalisasi wajib belum diterima.
                if now - self.t_start > 3.0 and now - self.last_warn > 3.0:
                    self.last_warn = now
                    self.get_logger().warn(
                        f'MENUNGGU pose asli ({ODOM_TOPIC} / {POSE_TOPIC}). Robot belum digerakkan. '
                        'Jalankan bridge pose (lihat langkah bridge), atau jalankan dengan '
                        '--ros-args -p require_gt:=false untuk dead reckoning.')
                return
            if self.last_t is None:
                return
            if not self.warned and now - self.t_start > 3.0:
                self.warned = True
                self.get_logger().warn(
                    'Lokalisasi referensi belum tersedia; posisi dihitung melalui estimasi putaran roda.')
        if not self.running:
            if self.have_odom and self.abs_source:
                d = math.hypot(self.x - START_POSE[0], self.y - START_POSE[1])
                dyaw = abs(norm(self.yaw - START_POSE[2]))
                if d > 0.4 or dyaw > 0.6:
                    if now - self.last_warn > 3.0:
                        self.last_warn = now
                        self.get_logger().warn(
                            f'Robot TIDAK di posisi awal (pose asli x={self.x:.2f} y={self.y:.2f} '
                            f'yaw={math.degrees(self.yaw):.0f}, seharusnya sekitar '
                            f'({START_POSE[0]:.1f}, {START_POSE[1]:.1f}), yaw 0). '
                            'Restart launch Gazebo dulu, lalu jalankan program lagi.')
                    return
            self.running = True
            self.get_logger().info(
                f'Mulai dari x={self.x:.2f} y={self.y:.2f} yaw={math.degrees(self.yaw):.0f}')
        v, w = self.ctrl.step(self.x, self.y, self.yaw)
        cmd_v, cmd_w = physical_to_cmd(v, w)
        msg = Twist()
        msg.linear.x = cmd_v
        msg.angular.z = cmd_w
        self.cmd_pub.publish(msg)

        if self.ctrl.idx != self.last_idx:
            self.last_idx = self.ctrl.idx
            if self.ctrl.done:
                self.get_logger().info(
                    f'Navigasi selesai. Posisi akhir x={self.x:.2f}, y={self.y:.2f}')
            else:
                self.get_logger().info(
                    f'Menuju waypoint {self.ctrl.idx + 1}/{len(WAYPOINTS)} | '
                    f'[{"POSE ASLI" if self.have_odom else "RODA"}] '
                    f'pose x={self.x:.2f} y={self.y:.2f} yaw={math.degrees(self.yaw):.0f}')


def main(args=None):
    rclpy.init(args=args)
    node = TrackRunner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()