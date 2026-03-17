from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copytree, rmtree
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
    source_mass_kg: float | None = None
    motor_constant_scale: float | None = None


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
        source_mass_kg=None if payload.get("source_mass_kg") is None else float(payload["source_mass_kg"]),
        motor_constant_scale=None
        if payload.get("motor_constant_scale") is None
        else float(payload["motor_constant_scale"]),
    )


def _base_link(model: ET.Element) -> ET.Element | None:
    base_link = model.find("./link[@name='base_link']")
    if base_link is None:
        base_link = model.find("./link")
    if base_link is None:
        base_link = model.find(".//link[@name='base_link']")
    if base_link is None:
        base_link = model.find(".//link")
    return base_link


def _apply_inertial_override(base_link: ET.Element, config: GazeboOverlayConfig) -> None:
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


def _motor_constant_scale(config: GazeboOverlayConfig) -> float:
    if config.motor_constant_scale is not None:
        return config.motor_constant_scale
    if config.source_mass_kg is None or config.source_mass_kg <= 0.0:
        return 1.0
    return config.mass_kg / config.source_mass_kg


def _apply_motor_model_override(model: ET.Element, config: GazeboOverlayConfig) -> None:
    scale = _motor_constant_scale(config)
    if abs(scale - 1.0) < 1.0e-9:
        return

    for plugin in model.findall("./plugin"):
        if plugin.get("name") != "gz::sim::systems::MulticopterMotorModel":
            continue
        motor_constant = plugin.find("motorConstant")
        if motor_constant is None or motor_constant.text is None:
            continue
        motor_constant_value = float(motor_constant.text)
        motor_constant.text = f"{motor_constant_value * scale:.8e}"


def _merged_include_uri(model: ET.Element) -> str | None:
    for include in model.findall("./include"):
        if include.get("merge", "").lower() != "true":
            continue
        uri = include.find("uri")
        if uri is None or uri.text is None:
            continue
        if uri.text.startswith("model://"):
            return uri.text.removeprefix("model://")
    return None


def _replace_merged_include_uri(model: ET.Element, target_model_name: str) -> bool:
    for include in model.findall("./include"):
        if include.get("merge", "").lower() != "true":
            continue
        uri = include.find("uri")
        if uri is None or uri.text is None:
            continue
        if uri.text.startswith("model://"):
            uri.text = f"model://{target_model_name}"
            return True
    return False


def _derived_target_name(config: GazeboOverlayConfig, source_name: str) -> str:
    prefix = f"{config.source_model_name}_"
    if source_name.startswith(prefix):
        suffix = source_name[len(config.source_model_name) :]
        return f"{config.target_model_name}{suffix}"
    return f"{config.target_model_name}_{source_name}"


def _rewrite_model_uri_references(xml_text: str, source_model_name: str, target_model_name: str) -> str:
    return xml_text.replace(f"model://{source_model_name}/", f"model://{target_model_name}/")


def patch_model_sdf(
    model_sdf_text: str,
    config: GazeboOverlayConfig,
    include_target_model_name: str | None = None,
) -> str:
    root = ET.fromstring(model_sdf_text)
    model = root.find(".//model")
    if model is None:
        raise ValueError("Could not find <model> element in model.sdf")
    model.set("name", config.target_model_name)
    _apply_motor_model_override(model, config)

    base_link = _base_link(model)
    if base_link is not None:
        _apply_inertial_override(base_link, config)
    elif include_target_model_name is not None and _replace_merged_include_uri(model, include_target_model_name):
        pass
    else:
        raise ValueError("Could not find a patchable <link> element or merged <include> in model.sdf")

    xml_text = ET.tostring(root, encoding="unicode")
    xml_text = _rewrite_model_uri_references(xml_text, config.source_model_name, config.target_model_name)
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
        rmtree(destination_dir)

    copytree(source_model_dir, destination_dir)

    model_sdf = destination_dir / "model.sdf"
    if not model_sdf.exists():
        raise FileNotFoundError(
            f"Expected {model_sdf}. The overlay installer currently requires a concrete model.sdf file."
        )
    model_sdf_text = model_sdf.read_text(encoding="utf-8")
    root = ET.fromstring(model_sdf_text)
    model = root.find(".//model")
    include_model_name = _merged_include_uri(model) if model is not None and _base_link(model) is None else None
    include_target_model_name = None
    if include_model_name is not None:
        include_source_dir = source_model_dir.parent / include_model_name
        if not include_source_dir.exists():
            raise FileNotFoundError(
                f"Expected merged include source model directory: {include_source_dir}"
            )
        include_target_model_name = _derived_target_name(config, include_model_name)
        include_config = GazeboOverlayConfig(
            source_model_name=include_model_name,
            target_model_name=include_target_model_name,
            mass_kg=config.mass_kg,
            ixx=config.ixx,
            iyy=config.iyy,
            izz=config.izz,
            source_mass_kg=config.source_mass_kg,
            motor_constant_scale=config.motor_constant_scale,
        )
        install_overlay(include_source_dir, destination_root, include_config)

    model_sdf.write_text(
        patch_model_sdf(model_sdf_text, config, include_target_model_name=include_target_model_name),
        encoding="utf-8",
    )

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
                f"source_mass_kg={config.source_mass_kg}",
                f"motor_constant_scale={_motor_constant_scale(config)}",
            ]
        ),
        encoding="utf-8",
    )
    return destination_dir
