from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copytree
from xml.etree import ElementTree as ET

import yaml


@dataclass(slots=True)
class GazeboOverlayConfig:
    source_model_name: str
    target_model_name: str
    mass_kg: float
    ixx: float
    iyy: float
    izz: float


def load_overlay_config(path: Path) -> GazeboOverlayConfig:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    inertia = payload.get("inertia", {})
    return GazeboOverlayConfig(
        source_model_name=payload["source_model_name"],
        target_model_name=payload["target_model_name"],
        mass_kg=float(payload["mass_kg"]),
        ixx=float(inertia["ixx"]),
        iyy=float(inertia["iyy"]),
        izz=float(inertia["izz"]),
    )


def patch_model_sdf(model_sdf_text: str, config: GazeboOverlayConfig) -> str:
    root = ET.fromstring(model_sdf_text)
    model = root.find(".//model")
    if model is None:
        raise ValueError("Could not find <model> element in model.sdf")
    model.set("name", config.target_model_name)

    base_link = model.find(".//link[@name='base_link']")
    if base_link is None:
        base_link = model.find(".//link")
    if base_link is None:
        raise ValueError("Could not find a <link> element in model.sdf")

    inertial = base_link.find("inertial")
    if inertial is None:
        inertial = ET.SubElement(base_link, "inertial")
    mass = inertial.find("mass")
    if mass is None:
        mass = ET.SubElement(inertial, "mass")
    mass.text = f"{config.mass_kg:.6f}"

    inertia = inertial.find("inertia")
    if inertia is None:
        inertia = ET.SubElement(inertial, "inertia")

    for tag, value in {
        "ixx": config.ixx,
        "iyy": config.iyy,
        "izz": config.izz,
        "ixy": 0.0,
        "ixz": 0.0,
        "iyz": 0.0,
    }.items():
        element = inertia.find(tag)
        if element is None:
            element = ET.SubElement(inertia, tag)
        element.text = f"{value:.6f}"

    xml_text = ET.tostring(root, encoding="unicode")
    return "<?xml version=\"1.0\" ?>\n" + xml_text


def patch_model_config(model_config_text: str, config: GazeboOverlayConfig) -> str:
    root = ET.fromstring(model_config_text)
    name = root.find("name")
    if name is None:
        name = ET.SubElement(root, "name")
    name.text = config.target_model_name
    description = root.find("description")
    if description is None:
        description = ET.SubElement(root, "description")
    description.text = (
        f"{config.target_model_name}: x500-based PX4 Gazebo model with MATLAB mass/inertia "
        "overrides for the quantized Koopman MPC workflow."
    )
    xml_text = ET.tostring(root, encoding="unicode")
    return "<?xml version=\"1.0\" ?>\n" + xml_text


def install_overlay(source_model_dir: Path, destination_root: Path, config: GazeboOverlayConfig) -> Path:
    if not source_model_dir.exists():
        raise FileNotFoundError(f"Source model directory does not exist: {source_model_dir}")

    destination_dir = destination_root / config.target_model_name
    if destination_dir.exists():
        for child in destination_dir.iterdir():
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file() or nested.is_symlink():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
        destination_dir.rmdir()

    copytree(source_model_dir, destination_dir)

    model_sdf = destination_dir / "model.sdf"
    if not model_sdf.exists():
        raise FileNotFoundError(
            f"Expected {model_sdf}. The overlay installer currently requires a concrete model.sdf file."
        )
    model_sdf.write_text(patch_model_sdf(model_sdf.read_text(encoding="utf-8"), config), encoding="utf-8")

    model_config = destination_dir / "model.config"
    if model_config.exists():
        model_config.write_text(
            patch_model_config(model_config.read_text(encoding="utf-8"), config),
            encoding="utf-8",
        )

    metadata = destination_dir / "QUANTIZED_KOOPMAN_MODEL.txt"
    metadata.write_text(
        "\n".join(
            [
                f"source_model_name={config.source_model_name}",
                f"target_model_name={config.target_model_name}",
                f"mass_kg={config.mass_kg}",
                f"ixx={config.ixx}",
                f"iyy={config.iyy}",
                f"izz={config.izz}",
            ]
        ),
        encoding="utf-8",
    )
    return destination_dir

