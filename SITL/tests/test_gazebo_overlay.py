from quantized_quadrotor_sitl.utils.gazebo_overlay import (
    GazeboOverlayConfig,
    install_overlay,
    patch_model_config,
    patch_model_sdf,
)


def test_patch_model_sdf_overrides_name_and_inertia():
    original = """
    <sdf version="1.8">
      <model name="x500">
        <link name="base_link">
          <inertial>
            <mass>2.0</mass>
            <inertia>
              <ixx>1</ixx>
              <iyy>1</iyy>
              <izz>1</izz>
            </inertia>
          </inertial>
        </link>
      </model>
    </sdf>
    """
    config = GazeboOverlayConfig("x500", "quantized_koopman_quad", 4.34, 0.082, 0.0845, 0.1377)
    patched = patch_model_sdf(original, config)
    assert "quantized_koopman_quad" in patched
    assert "<mass>4.340000</mass>" in patched
    assert "<ixx>0.082000</ixx>" in patched
    assert "<iyy>0.084500</iyy>" in patched
    assert "<izz>0.137700</izz>" in patched


def test_patch_model_config_updates_name():
    original = """
    <model>
      <name>x500</name>
      <description>original</description>
    </model>
    """
    config = GazeboOverlayConfig("x500", "quantized_koopman_quad", 4.34, 0.082, 0.0845, 0.1377)
    patched = patch_model_config(original, config)
    assert "<name>quantized_koopman_quad</name>" in patched
    assert "MATLAB mass/inertia overrides" in patched


def test_install_overlay_patches_merged_base_model(tmp_path):
    source_root = tmp_path / "models"
    x500_dir = source_root / "x500"
    x500_dir.mkdir(parents=True)
    x500_dir.joinpath("model.sdf").write_text(
        """
        <sdf version="1.9">
          <model name="x500">
            <include merge="true">
              <uri>model://x500_base</uri>
            </include>
          </model>
        </sdf>
        """,
        encoding="utf-8",
    )
    x500_dir.joinpath("model.config").write_text(
        "<model><name>x500</name><description>top level</description></model>",
        encoding="utf-8",
    )

    x500_base_dir = source_root / "x500_base"
    x500_base_dir.mkdir()
    x500_base_dir.joinpath("model.sdf").write_text(
        """
        <sdf version="1.9">
          <model name="x500_base">
            <link name="base_link">
              <inertial>
                <mass>2.0</mass>
                <inertia>
                  <ixx>1.0</ixx>
                  <iyy>1.0</iyy>
                  <izz>1.0</izz>
                </inertia>
              </inertial>
            </link>
          </model>
        </sdf>
        """,
        encoding="utf-8",
    )
    x500_base_dir.joinpath("model.config").write_text(
        "<model><name>x500_base</name><description>base</description></model>",
        encoding="utf-8",
    )

    destination_root = tmp_path / "generated"
    config = GazeboOverlayConfig("x500", "quantized_koopman_quad", 4.34, 0.082, 0.0845, 0.1377)
    install_overlay(x500_dir, destination_root, config)

    top_level_sdf = (destination_root / "quantized_koopman_quad" / "model.sdf").read_text(encoding="utf-8")
    assert "model://quantized_koopman_quad_base" in top_level_sdf
    assert 'model name="quantized_koopman_quad"' in top_level_sdf

    base_sdf = (destination_root / "quantized_koopman_quad_base" / "model.sdf").read_text(encoding="utf-8")
    assert 'model name="quantized_koopman_quad_base"' in base_sdf
    assert "<mass>4.340000</mass>" in base_sdf
    assert "<ixx>0.082000</ixx>" in base_sdf
