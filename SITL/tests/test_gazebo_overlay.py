from quantized_quadrotor_sitl.utils.gazebo_overlay import GazeboOverlayConfig, patch_model_config, patch_model_sdf


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
