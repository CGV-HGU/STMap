import math

import cv2
import numpy as np


class GraphVisualizer:
    def __init__(self, map_size=512, scale=40.0):
        self.map_size = map_size
        self.scale = scale
        self.center_x = map_size // 2
        self.center_y = map_size // 2
        self.origin_x = self.center_x
        self.origin_y = self.center_y

    def world_to_pixel(self, x, y):
        px = int(self.origin_x + x * self.scale)
        py = int(self.origin_y - y * self.scale)
        return px, py

    def update_camera(self, cx, cy):
        self.origin_x = self.center_x - cx * self.scale
        self.origin_y = self.center_y + cy * self.scale

    def draw_map(self, manager):
        rx, ry, ryaw = manager.get_current_pose()
        self.update_camera(rx, ry)

        bg_color = (20, 20, 26)
        canvas = np.full((self.map_size, self.map_size, 3), bg_color, dtype=np.uint8)

        grid_color = (35, 35, 45)
        meters_half = (self.map_size // 2) / self.scale
        start_x = int(rx - meters_half) - 1
        end_x = int(rx + meters_half) + 1
        start_y = int(ry - meters_half) - 1
        end_y = int(ry + meters_half) + 1

        for i in range(start_x, end_x):
            px, _ = self.world_to_pixel(i, 0)
            if 0 <= px < self.map_size:
                cv2.line(canvas, (px, 0), (px, self.map_size), grid_color, 1, cv2.LINE_AA)
        for i in range(start_y, end_y):
            _, py = self.world_to_pixel(0, i)
            if 0 <= py < self.map_size:
                cv2.line(canvas, (0, py), (self.map_size, py), grid_color, 1, cv2.LINE_AA)

        for edge in manager.edges:
            uid, vid = edge.get("from"), edge.get("to")
            if uid in manager.anchors and vid in manager.anchors:
                u_node = manager.anchors[uid]
                v_node = manager.anchors[vid]
                ux, uy = self.world_to_pixel(u_node.pose[0], u_node.pose[1])
                vx, vy = self.world_to_pixel(v_node.pose[0], v_node.pose[1])

                edge_color = (90, 90, 110)
                u_place = manager.places.get(u_node.place_id)
                if u_place:
                    ptype = u_place.place_type.lower()
                    if "corridor" in ptype or "hallway" in ptype:
                        edge_color = (170, 150, 95)
                    else:
                        edge_color = (120, 110, 160)
                if edge.get("type") == "transition":
                    edge_color = (70, 190, 230)

                cv2.line(canvas, (ux, uy), (vx, vy), edge_color, 2, cv2.LINE_AA)

        for aid, anchor in manager.anchors.items():
            px, py = self.world_to_pixel(anchor.pose[0], anchor.pose[1])
            parent_place = manager.places.get(anchor.place_id)
            if not parent_place:
                continue

            place_type = parent_place.place_type.lower()
            color = (200, 200, 210)
            outline = (235, 235, 245)
            radius = 4

            if anchor.is_door:
                color = (50, 220, 255)
                outline = (120, 255, 255)
            elif "corridor" in place_type or "hallway" in place_type:
                color = (170, 150, 95)
                outline = (210, 185, 120)
            else:
                color = (120, 110, 160)
                outline = (175, 165, 210)
                radius = 5

            if anchor.is_door:
                size = 6
                pts_outline = np.array(
                    [[px, py - size - 1], [px - size - 1, py + size + 1], [px + size + 1, py + size + 1]],
                    np.int32,
                ).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts_outline], True, outline, 2, cv2.LINE_AA)
                pts = np.array(
                    [[px, py - size], [px - size, py + size], [px + size, py + size]],
                    np.int32,
                ).reshape((-1, 1, 2))
                cv2.fillPoly(canvas, [pts], color)
            elif "corridor" not in place_type and "hallway" not in place_type:
                size = 5
                cv2.rectangle(
                    canvas,
                    (px - size - 1, py - size - 1),
                    (px + size + 1, py + size + 1),
                    outline,
                    2,
                    cv2.LINE_AA,
                )
                cv2.rectangle(canvas, (px - size, py - size), (px + size, py + size), color, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (px, py), radius + 1, outline, 2, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), radius, color, -1, cv2.LINE_AA)

        for pid, place in manager.places.items():
            if not place.anchors:
                continue
            sum_x = 0.0
            sum_y = 0.0
            count = 0
            for aid in place.anchors:
                if aid in manager.anchors:
                    a = manager.anchors[aid]
                    sum_x += a.pose[0]
                    sum_y += a.pose[1]
                    count += 1
            if count == 0:
                continue
            cx, cy = sum_x / count, sum_y / count
            cpx, cpy = self.world_to_pixel(cx, cy)
            label = place.place_type.title()
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (cpx - 4, cpy - th - 6), (cpx + tw + 4, cpy + 4), (10, 10, 14), -1)
            cv2.rectangle(canvas, (cpx - 4, cpy - th - 6), (cpx + tw + 4, cpy + 4), (70, 70, 90), 1)
            cv2.putText(canvas, label, (cpx, cpy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 240), 1, cv2.LINE_AA)

        rpx, rpy = self.world_to_pixel(rx, ry)
        cv2.circle(canvas, (rpx, rpy), 8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (rpx, rpy), 5, (60, 160, 255), -1, cv2.LINE_AA)
        arrow_len = 16
        tip_x = int(rpx + arrow_len * math.cos(ryaw))
        tip_y = int(rpy - arrow_len * math.sin(ryaw))
        cv2.arrowedLine(canvas, (rpx, rpy), (tip_x, tip_y), (255, 255, 255), 2, cv2.LINE_AA, 0, 0.4)

        anchors_count = len(manager.anchors)
        places_count = len(manager.places)
        edges_count = len(manager.edges)
        current_place = manager.current_place.place_type if manager.current_place else "none"
        info = f"A:{anchors_count} E:{edges_count} P:{places_count} Cur:{current_place}"
        (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (8, 8), (8 + tw + 8, 8 + th + 8), (10, 10, 14), -1)
        cv2.rectangle(canvas, (8, 8), (8 + tw + 8, 8 + th + 8), (70, 70, 90), 1)
        cv2.putText(canvas, info, (12, 8 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 240), 1, cv2.LINE_AA)

        return canvas
