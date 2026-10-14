#!/usr/bin/env python3
"""Convert a GMR X1 motion pickle file to CSV.

The DOF column order is read from the X1 MJCF description file, using the
same depth-first order as GMR's KinematicsModel.  The output layout is:

    root_x, root_y, root_z,
    root_qx, root_qy, root_qz, root_qw,
    <X1 joint names ...>

Usage examples:
    python pkl_to_csv_x1.py
    python pkl_to_csv_x1.py --pkl motion/x1/pkl/Leo_v2.pkl
    python pkl_to_csv_x1.py --pkl input.pkl --csv output.csv
"""

from __future__ import annotations

import argparse
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
X1_XML = SCRIPT_DIR / "assets/x1/mjcf/robot/xyber_x1/xyber_x1_serial.xml"
DEFAULT_PKL = SCRIPT_DIR / "motion/x1/pkl/Leo.pkl"

ROOT_COLUMNS = [
    "root_x",
    "root_y",
    "root_z",
    "root_qx",
    "root_qy",
    "root_qz",
    "root_qw",
]


def get_x1_joint_names(xml_file: Path) -> list[str]:
    """Read X1 joint names in GMR's MJCF traversal order."""
    root = ET.parse(xml_file).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"<worldbody> not found in {xml_file}")

    robot_root = worldbody.find("body")
    if robot_root is None:
        raise ValueError(f"Robot root body not found in {xml_file}")

    joint_names: list[str] = []

    def visit(body: ET.Element) -> None:
        # This is the same order used by KinematicsModel._add_xml_body:
        # process the current body's joint, then recursively process children.
        for joint in body.findall("joint"):
            name = joint.get("name")
            if name:
                joint_names.append(name)
        for child in body.findall("body"):
            visit(child)

    visit(robot_root)
    if not joint_names:
        raise ValueError(f"No joints found in {xml_file}")
    return joint_names


def load_motion(pkl_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with pkl_file.open("rb") as file:
        motion_data = pickle.load(file)

    required_keys = {"root_pos", "root_rot", "dof_pos"}
    missing = required_keys.difference(motion_data)
    if missing:
        raise KeyError(f"Missing keys in {pkl_file}: {sorted(missing)}")

    root_pos = np.asarray(motion_data["root_pos"])
    root_rot = np.asarray(motion_data["root_rot"])
    dof_pos = np.asarray(motion_data["dof_pos"])

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape (N, 3), got {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"root_rot must have shape (N, 4), got {root_rot.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"dof_pos must have shape (N, D), got {dof_pos.shape}")

    frame_count = root_pos.shape[0]
    if root_rot.shape[0] != frame_count or dof_pos.shape[0] != frame_count:
        raise ValueError(
            "root_pos, root_rot, and dof_pos must contain the same number of frames: "
            f"{root_pos.shape[0]}, {root_rot.shape[0]}, {dof_pos.shape[0]}"
        )

    return root_pos, root_rot, dof_pos


def convert(pkl_file: Path, csv_file: Path, xml_file: Path) -> None:
    joint_names = get_x1_joint_names(xml_file)
    root_pos, root_rot, dof_pos = load_motion(pkl_file)

    if dof_pos.shape[1] != len(joint_names):
        raise ValueError(
            f"X1 dof_pos has {dof_pos.shape[1]} columns, but the MJCF defines "
            f"{len(joint_names)} joints.\n"
            f"MJCF: {xml_file}"
        )

    data = np.concatenate((root_pos, root_rot, dof_pos), axis=1)
    columns = ROOT_COLUMNS + joint_names

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        csv_file,
        data,
        delimiter=",",
        header=",".join(columns),
        comments="",
    )

    print(f"data shape: {data.shape}")
    print(f"number of columns: {len(columns)}")
    print(f"CSV saved to: {csv_file}")
    print(f"X1 DOF count: {len(joint_names)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a GMR X1 pickle motion file to a named CSV file."
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=DEFAULT_PKL,
        help=f"Input GMR pickle file (default: {DEFAULT_PKL})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV file (default: same path as --pkl with a .csv suffix)",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=X1_XML,
        help=f"X1 MJCF description file (default: {X1_XML})",
    )
    args = parser.parse_args()

    pkl_file = args.pkl.expanduser().resolve()
    xml_file = args.xml.expanduser().resolve()
    csv_file = (
        args.csv.expanduser().resolve()
        if args.csv is not None
        else pkl_file.with_suffix(".csv")
    )

    if not pkl_file.is_file():
        raise FileNotFoundError(f"Input pickle file not found: {pkl_file}")
    if not xml_file.is_file():
        raise FileNotFoundError(f"X1 MJCF file not found: {xml_file}")

    convert(pkl_file, csv_file, xml_file)


if __name__ == "__main__":
    main()
