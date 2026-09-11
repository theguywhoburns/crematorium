{ pkgs, lib, config, inputs, ... }:

{
  packages = [
    pkgs.git
    pkgs.openssl
    pkgs.cudaPackages.cudatoolkit
    pkgs.pyright
  ];
  languages.python = {
    enable = true;
    version = "3.12";
    venv.enable = true;
    uv = {
      enable = true;
      sync = {
        enable = true;
        # Match the CUDA 13.0 build declared in pyproject.toml. Switch to a
        # different backend by changing this extra (see pyproject [project.optional-dependencies]).
        extras = [ "cu130" ];
      };
    };
  };
  enterShell = ''
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH
    export CUDA_HOME="${pkgs.cudaPackages.cudatoolkit}"
    export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
    export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
    source .devenv/state/venv/bin/activate
  '';

  # https://devenv.sh/tests/
  enterTest = ''
    # TODO: setup tests
  '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
