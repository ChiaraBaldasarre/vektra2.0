"""
SVG Parsing Module for Vektra.
This module extracts 2D paths/polygons from SVG elements,
handles nested transformations (matrices, translation, rotation, scale),
and discretizes Bezier curves into line segments.
"""

import io
import re
import math
import xml.etree.ElementTree as ET
# pyrefly: ignore [missing-import]
import numpy as np


def multiply_matrices(m1, m2):
    """
    Multiplies two 2D affine transformation matrices.
    Each matrix is represented as (a, b, c, d, e, f).
    """
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1
    )


def apply_transform(point, matrix):
    """
    Applies a 2D affine transformation matrix to a point.
    """
    x, y = point
    a, b, c, d, e, f = matrix
    return (
        a * x + c * y + e,
        b * x + d * y + f
    )


def parse_transform(transform_str):
    """
    Parses an SVG transform attribute string into a single cumulative affine matrix.
    Supports matrix, translate, scale, and rotate.
    """
    if not transform_str:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)  # Identity matrix

    curr = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    # Find all commands like translate(...) or matrix(...)
    pattern = r'(\w+)\s*\(([^)]+)\)'
    matches = re.findall(pattern, transform_str)

    for name, args_str in matches:
        args = [float(x) for x in re.split(r'[,\s]+', args_str.strip()) if x]
        if name == 'matrix' and len(args) == 6:
            m = tuple(args)
        elif name == 'translate':
            tx = args[0]
            ty = args[1] if len(args) > 1 else 0.0
            m = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == 'scale':
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            m = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == 'rotate':
            angle_deg = args[0]
            angle = math.radians(angle_deg)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            if len(args) == 3:
                cx, cy = args[1], args[2]
                m1 = (1.0, 0.0, 0.0, 1.0, cx, cy)
                m2 = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
                m3 = (1.0, 0.0, 0.0, 1.0, -cx, -cy)
                m = multiply_matrices(m1, multiply_matrices(m2, m3))
            else:
                m = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
        else:
            m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        
        curr = multiply_matrices(curr, m)
    
    return curr


def evaluate_bezier_cubic(p0, p1, p2, p3, t):
    """
    Evaluates a cubic Bezier curve at parameter t [0, 1].
    """
    one_minus_t = 1.0 - t
    return (
        one_minus_t**3 * p0[0] + 3 * one_minus_t**2 * t * p1[0] + 3 * one_minus_t * t**2 * p2[0] + t**3 * p3[0],
        one_minus_t**3 * p0[1] + 3 * one_minus_t**2 * t * p1[1] + 3 * one_minus_t * t**2 * p2[1] + t**3 * p3[1]
    )


def evaluate_bezier_quadratic(p0, p1, p2, t):
    """
    Evaluates a quadratic Bezier curve at parameter t [0, 1].
    """
    one_minus_t = 1.0 - t
    return (
        one_minus_t**2 * p0[0] + 2 * one_minus_t * t * p1[0] + t**2 * p2[0],
        one_minus_t**2 * p0[1] + 2 * one_minus_t * t * p1[1] + t**2 * p2[1]
    )


class SVGParser:
    """
    Main SVG parsing engine.
    Parses SVG files and yields discretized 2D paths/contours.
    """
    def __init__(self, bezier_resolution=30):
        self.bezier_resolution = bezier_resolution

    def parse_file(self, file_source):
        """
        Parses an SVG file and returns a list of dictionaries,
        each containing element type, points, and whether the path is closed.
        Accepts a file path string, a BytesIO object, or a Streamlit UploadedFile.
        """
        # Streamlit UploadedFile and BytesIO both support .read()
        if hasattr(file_source, 'read'):
            raw_bytes = file_source.read()
            tree = ET.parse(io.BytesIO(raw_bytes))
        else:
            tree = ET.parse(file_source)
        root = tree.getroot()
        
        # XML namespace resolution
        ns = ""
        m = re.match(r'({.*})', root.tag)
        if m:
            ns = m.group(1)
            
        elements = []
        identity = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        self._traverse_node(root, ns, identity, elements)
        return elements

    def _traverse_node(self, node, ns, parent_matrix, elements):
        """
        Recursively traverses SVG nodes to accumulate transformations
        and extract leaf geometry elements.
        """
        # Parse local transform
        local_transform = parse_transform(node.attrib.get('transform', ''))
        current_matrix = multiply_matrices(parent_matrix, local_transform)
        
        # Local tag name (remove namespace)
        tag = node.tag.replace(ns, "") if ns else node.tag
        
        if tag in ('g', 'svg'):
            # Both group elements and the SVG root traverse their children
            for child in node:
                self._traverse_node(child, ns, current_matrix, elements)
        
        elif tag == 'rect':
            x = float(node.attrib.get('x', 0.0))
            y = float(node.attrib.get('y', 0.0))
            width = float(node.attrib.get('width', 0.0))
            height = float(node.attrib.get('height', 0.0))
            
            points = [
                (x, y),
                (x + width, y),
                (x + width, y + height),
                (x, y + height)
            ]
            transformed_points = [apply_transform(pt, current_matrix) for pt in points]
            elements.append({
                'type': 'rect',
                'points': np.array(transformed_points, dtype=np.float32),
                'closed': True
            })
            
        elif tag == 'circle':
            cx = float(node.attrib.get('cx', 0.0))
            cy = float(node.attrib.get('cy', 0.0))
            r = float(node.attrib.get('r', 0.0))
            
            # Discretize circle into 64 steps
            points = []
            steps = 64
            for i in range(steps):
                angle = 2 * math.pi * i / steps
                px = cx + r * math.cos(angle)
                py = cy + r * math.sin(angle)
                points.append((px, py))
                
            transformed_points = [apply_transform(pt, current_matrix) for pt in points]
            elements.append({
                'type': 'circle',
                'points': np.array(transformed_points, dtype=np.float32),
                'closed': True
            })

        elif tag == 'ellipse':
            cx = float(node.attrib.get('cx', 0.0))
            cy = float(node.attrib.get('cy', 0.0))
            rx = float(node.attrib.get('rx', 0.0))
            ry = float(node.attrib.get('ry', 0.0))

            # Discretize ellipse into 64 steps
            points = []
            steps = 64
            for i in range(steps):
                angle = 2 * math.pi * i / steps
                px = cx + rx * math.cos(angle)
                py = cy + ry * math.sin(angle)
                points.append((px, py))

            transformed_points = [apply_transform(pt, current_matrix) for pt in points]
            elements.append({
                'type': 'ellipse',
                'points': np.array(transformed_points, dtype=np.float32),
                'closed': True
            })

        elif tag == 'line':
            x1 = float(node.attrib.get('x1', 0.0))
            y1 = float(node.attrib.get('y1', 0.0))
            x2 = float(node.attrib.get('x2', 0.0))
            y2 = float(node.attrib.get('y2', 0.0))
            points = [(x1, y1), (x2, y2)]
            transformed_points = [apply_transform(pt, current_matrix) for pt in points]
            elements.append({
                'type': 'line',
                'points': np.array(transformed_points, dtype=np.float32),
                'closed': False
            })

        elif tag == 'polygon' or tag == 'polyline':

            points_str = node.attrib.get('points', '').strip()
            if points_str:
                coords = [float(x) for x in re.split(r'[,\s]+', points_str) if x]
                points = [(coords[i], coords[i+1]) for i in range(0, len(coords) - 1, 2)]
                transformed_points = [apply_transform(pt, current_matrix) for pt in points]
                elements.append({
                    'type': tag,
                    'points': np.array(transformed_points, dtype=np.float32),
                    'closed': (tag == 'polygon')
                })
                
        elif tag == 'path':
            d = node.attrib.get('d', '')
            if d:
                parsed_paths = self._parse_path_d(d, current_matrix)
                elements.extend(parsed_paths)
                
        # Note: 'svg' and 'g' tags are already handled together at the top of this method.

    def _parse_path_d(self, d, matrix):
        """
        Parses path 'd' string attribute using regex tokenization.
        Discretizes into list of individual subpaths.
        Supports: M, m, L, l, H, h, V, v, C, c, S, s, Q, q, T, t, A, a, Z, z
        """
        # Robust SVG path tokenizer:
        # - Captures command letters (excluding e/E which are exponent notation)
        # - Captures numbers including scientific notation
        # - Handles implicit separation where negative sign starts a new number
        token_pattern = re.compile(
            r'([MmZzLlHhVvCcSsQqTtAa])'          # SVG command letters
            r'|([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'  # Numbers (int/float/scientific)
        )
        tokens = []
        for match in token_pattern.finditer(d):
            cmd_tok, val_tok = match.groups()
            if cmd_tok:
                tokens.append(cmd_tok)
            elif val_tok:
                tokens.append(float(val_tok))

        subpaths = []
        current_points = []
        closed = False

        cursor = (0.0, 0.0)
        start_point = (0.0, 0.0)
        last_control_cubic = (0.0, 0.0)
        last_control_quad = (0.0, 0.0)

        idx = 0
        n_tokens = len(tokens)
        cmd = None

        def get_args(count):
            nonlocal idx
            args = []
            for _ in range(count):
                if idx < n_tokens and isinstance(tokens[idx], (int, float)):
                    args.append(tokens[idx])
                    idx += 1
                else:
                    break
            return args

        def get_flag():
            """Read a single arc flag (0 or 1) from the token stream."""
            nonlocal idx
            if idx < n_tokens and isinstance(tokens[idx], (int, float)):
                val = tokens[idx]
                idx += 1
                return int(val)
            return 0

        def discretize_arc(cx, cy, rx, ry, x_rotation_deg, angle_start_deg, angle_sweep_deg, steps=None):
            """Discretize an SVG elliptical arc into line points."""
            if steps is None:
                steps = max(8, int(abs(angle_sweep_deg) / 5))
            x_rot = math.radians(x_rotation_deg)
            cos_rot = math.cos(x_rot)
            sin_rot = math.sin(x_rot)
            pts = []
            for i in range(1, steps + 1):
                angle = math.radians(angle_start_deg + angle_sweep_deg * i / steps)
                px = cx + rx * math.cos(angle) * cos_rot - ry * math.sin(angle) * sin_rot
                py = cy + rx * math.cos(angle) * sin_rot + ry * math.sin(angle) * cos_rot
                pts.append((px, py))
            return pts

        def endpoint_to_center_arc(x1, y1, x2, y2, fa, fs, rx, ry, phi_deg):
            """
            Convert SVG arc endpoint parameterization to center parameterization.
            Returns (cx, cy, rx, ry, angle_start_deg, angle_sweep_deg).
            Based on SVG spec section F.6.5.
            """
            phi = math.radians(phi_deg)
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)

            # Step 1
            dx2 = (x1 - x2) / 2.0
            dy2 = (y1 - y2) / 2.0
            x1p = cos_phi * dx2 + sin_phi * dy2
            y1p = -sin_phi * dx2 + cos_phi * dy2

            # Step 2 — correct radii
            rx = abs(rx)
            ry = abs(ry)
            x1p_sq = x1p * x1p
            y1p_sq = y1p * y1p
            rx_sq = rx * rx
            ry_sq = ry * ry

            # Ensure radii are large enough
            lamda = x1p_sq / rx_sq + y1p_sq / ry_sq
            if lamda > 1:
                sqrt_l = math.sqrt(lamda)
                rx = sqrt_l * rx
                ry = sqrt_l * ry
                rx_sq = rx * rx
                ry_sq = ry * ry

            num = max(0.0, rx_sq * ry_sq - rx_sq * y1p_sq - ry_sq * x1p_sq)
            den = rx_sq * y1p_sq + ry_sq * x1p_sq
            sq = math.sqrt(num / den) if den != 0 else 0.0
            if fa == fs:
                sq = -sq

            cxp = sq * rx * y1p / ry
            cyp = -sq * ry * x1p / rx

            # Step 3 — center
            cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
            cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

            # Step 4 — angles
            def angle_between(ux, uy, vx, vy):
                n = math.sqrt(ux * ux + uy * uy) * math.sqrt(vx * vx + vy * vy)
                if n == 0:
                    return 0.0
                c = max(-1.0, min(1.0, (ux * vx + uy * vy) / n))
                a = math.degrees(math.acos(c))
                if ux * vy - uy * vx < 0:
                    a = -a
                return a

            theta1 = angle_between(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
            d_theta = angle_between(
                (x1p - cxp) / rx, (y1p - cyp) / ry,
                (-x1p - cxp) / rx, (-y1p - cyp) / ry
            )

            if not fs and d_theta > 0:
                d_theta -= 360.0
            elif fs and d_theta < 0:
                d_theta += 360.0

            return cx, cy, rx, ry, theta1, d_theta

        while idx < n_tokens:
            token = tokens[idx]
            if isinstance(token, str):
                cmd = token
                idx += 1
            else:
                # Implicit command repetition
                if cmd == 'M':
                    cmd = 'L'
                elif cmd == 'm':
                    cmd = 'l'

            if cmd == 'M':
                args = get_args(2)
                if len(args) == 2:
                    cursor = (args[0], args[1])
                    start_point = cursor
                    if current_points:
                        subpaths.append({
                            'type': 'path',
                            'points': np.array([apply_transform(pt, matrix) for pt in current_points], dtype=np.float32),
                            'closed': closed
                        })
                        current_points = []
                        closed = False
                    current_points.append(cursor)
            elif cmd == 'm':
                args = get_args(2)
                if len(args) == 2:
                    cursor = (cursor[0] + args[0], cursor[1] + args[1])
                    start_point = cursor
                    if current_points:
                        subpaths.append({
                            'type': 'path',
                            'points': np.array([apply_transform(pt, matrix) for pt in current_points], dtype=np.float32),
                            'closed': closed
                        })
                        current_points = []
                        closed = False
                    current_points.append(cursor)

            elif cmd == 'L':
                args = get_args(2)
                if len(args) == 2:
                    cursor = (args[0], args[1])
                    current_points.append(cursor)
            elif cmd == 'l':
                args = get_args(2)
                if len(args) == 2:
                    cursor = (cursor[0] + args[0], cursor[1] + args[1])
                    current_points.append(cursor)

            elif cmd == 'H':
                args = get_args(1)
                if len(args) == 1:
                    cursor = (args[0], cursor[1])
                    current_points.append(cursor)
            elif cmd == 'h':
                args = get_args(1)
                if len(args) == 1:
                    cursor = (cursor[0] + args[0], cursor[1])
                    current_points.append(cursor)

            elif cmd == 'V':
                args = get_args(1)
                if len(args) == 1:
                    cursor = (cursor[0], args[0])
                    current_points.append(cursor)
            elif cmd == 'v':
                args = get_args(1)
                if len(args) == 1:
                    cursor = (cursor[0], cursor[1] + args[0])
                    current_points.append(cursor)

            elif cmd == 'C':
                args = get_args(6)
                if len(args) == 6:
                    p1 = (args[0], args[1])
                    p2 = (args[2], args[3])
                    p3 = (args[4], args[5])
                    for step in range(1, self.bezier_resolution + 1):
                        t = step / self.bezier_resolution
                        pt = evaluate_bezier_cubic(cursor, p1, p2, p3, t)
                        current_points.append(pt)
                    cursor = p3
                    last_control_cubic = p2
            elif cmd == 'c':
                args = get_args(6)
                if len(args) == 6:
                    p1 = (cursor[0] + args[0], cursor[1] + args[1])
                    p2 = (cursor[0] + args[2], cursor[1] + args[3])
                    p3 = (cursor[0] + args[4], cursor[1] + args[5])
                    for step in range(1, self.bezier_resolution + 1):
                        t = step / self.bezier_resolution
                        pt = evaluate_bezier_cubic(cursor, p1, p2, p3, t)
                        current_points.append(pt)
                    cursor = p3
                    last_control_cubic = p2

            elif cmd == 'S':
                args = get_args(4)
                if len(args) == 4:
                    p1 = (2 * cursor[0] - last_control_cubic[0], 2 * cursor[1] - last_control_cubic[1])
                    p2 = (args[0], args[1])
                    p3 = (args[2], args[3])
                    for step in range(1, self.bezier_resolution + 1):
                        t = step / self.bezier_resolution
                        pt = evaluate_bezier_cubic(cursor, p1, p2, p3, t)
                        current_points.append(pt)
                    cursor = p3
                    last_control_cubic = p2
            elif cmd == 's':
                args = get_args(4)
                if len(args) == 4:
                    p1 = (2 * cursor[0] - last_control_cubic[0], 2 * cursor[1] - last_control_cubic[1])
                    p2 = (cursor[0] + args[0], cursor[1] + args[1])
                    p3 = (cursor[0] + args[2], cursor[1] + args[3])
                    for step in range(1, self.bezier_resolution + 1):
                        t = step / self.bezier_resolution
                        pt = evaluate_bezier_cubic(cursor, p1, p2, p3, t)
                        current_points.append(pt)
                    cursor = p3
                    last_control_cubic = p2

            elif cmd == 'Q':
                args = get_args(4)
                if len(args) == 4:
                    p1 = (args[0], args[1])
                    p2 = (args[2], args[3])
                    for step in range(1, self.bezier_resolution + 1):
                        t = step / self.bezier_resolution
                        pt = evaluate_bezier_quadratic(cursor, p1, p2, t)
                        current_points.append(pt)
                    cursor = p2
                    last_control_quad = p1
            elif cmd == 'q':
                args = get_args(4)
                if len(args) == 4:
                    p1 = (cursor[0] + args[0], cursor[1] + args[1])
                    p2 = (cursor[0] + args[2], cursor[1] + args[3])
                    for step in range(1, self.bezier_resolution + 1):
                        t = step / self.bezier_resolution
                        pt = evaluate_bezier_quadratic(cursor, p1, p2, t)
                        current_points.append(pt)
                    cursor = p2
                    last_control_quad = p1

            elif cmd == 'T':
                args = get_args(2)
                if len(args) == 2:
                    p1 = (2 * cursor[0] - last_control_quad[0], 2 * cursor[1] - last_control_quad[1])
                    p2 = (args[0], args[1])
                    for step in range(1, self.bezier_resolution + 1):
                        t = step / self.bezier_resolution
                        pt = evaluate_bezier_quadratic(cursor, p1, p2, t)
                        current_points.append(pt)
                    cursor = p2
                    last_control_quad = p1
            elif cmd == 't':
                args = get_args(2)
                if len(args) == 2:
                    p1 = (2 * cursor[0] - last_control_quad[0], 2 * cursor[1] - last_control_quad[1])
                    p2 = (cursor[0] + args[0], cursor[1] + args[1])
                    for step in range(1, self.bezier_resolution + 1):
                        t = step / self.bezier_resolution
                        pt = evaluate_bezier_quadratic(cursor, p1, p2, t)
                        current_points.append(pt)
                    cursor = p2
                    last_control_quad = p1

            elif cmd == 'A':
                # Absolute elliptical arc: rx ry x-rot large-arc-flag sweep-flag x y
                args = get_args(3)   # rx, ry, x-rotation
                if len(args) == 3:
                    rx, ry, phi = args[0], args[1], args[2]
                    fa = get_flag()  # large-arc-flag (0 or 1)
                    fs = get_flag()  # sweep-flag (0 or 1)
                    end_args = get_args(2)
                    if len(end_args) == 2:
                        x2, y2 = end_args[0], end_args[1]
                        x1, y1 = cursor
                        if rx != 0 and ry != 0 and (x1, y1) != (x2, y2):
                            cx, cy, rx_c, ry_c, theta1, d_theta = endpoint_to_center_arc(
                                x1, y1, x2, y2, fa, fs, rx, ry, phi
                            )
                            arc_pts = discretize_arc(cx, cy, rx_c, ry_c, phi, theta1, d_theta)
                            current_points.extend(arc_pts)
                        cursor = (x2, y2)
                        if not current_points or current_points[-1] != cursor:
                            current_points.append(cursor)

            elif cmd == 'a':
                # Relative elliptical arc
                args = get_args(3)
                if len(args) == 3:
                    rx, ry, phi = args[0], args[1], args[2]
                    fa = get_flag()
                    fs = get_flag()
                    end_args = get_args(2)
                    if len(end_args) == 2:
                        x2 = cursor[0] + end_args[0]
                        y2 = cursor[1] + end_args[1]
                        x1, y1 = cursor
                        if rx != 0 and ry != 0 and (x1, y1) != (x2, y2):
                            cx, cy, rx_c, ry_c, theta1, d_theta = endpoint_to_center_arc(
                                x1, y1, x2, y2, fa, fs, rx, ry, phi
                            )
                            arc_pts = discretize_arc(cx, cy, rx_c, ry_c, phi, theta1, d_theta)
                            current_points.extend(arc_pts)
                        cursor = (x2, y2)
                        if not current_points or current_points[-1] != cursor:
                            current_points.append(cursor)

            elif cmd in ('Z', 'z'):
                closed = True
                if current_points and current_points[0] != cursor:
                    current_points.append(start_point)
                cursor = start_point
                subpaths.append({
                    'type': 'path',
                    'points': np.array([apply_transform(pt, matrix) for pt in current_points], dtype=np.float32),
                    'closed': closed
                })
                current_points = []
                closed = False

        # Save any remaining open subpath
        if current_points:
            subpaths.append({
                'type': 'path',
                'points': np.array([apply_transform(pt, matrix) for pt in current_points], dtype=np.float32),
                'closed': closed
            })

        return subpaths


