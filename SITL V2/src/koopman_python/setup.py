from distutils.core import setup

from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=[
        "koopman_python",
        "koopman_python.dynamics",
        "koopman_python.training",
        "koopman_python.edmd",
        "koopman_python.mpc",
        "koopman_python.quantization",
        "koopman_python.experiments",
    ],
    package_dir={"": "."},
)

setup(**setup_args)
