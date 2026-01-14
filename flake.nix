{
  description = "Whispy";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {
    self,
    nixpkgs,
  }: let
    forAllSystems = nixpkgs.lib.genAttrs ["x86_64-linux" "aarch64-linux"];
    configuredNixpkgs = forAllSystems (system:
      import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      });
    mkPackage = system: let
      pkgs = configuredNixpkgs.${system};
      python3 = pkgs.python3;
      python3Packages = python3.pkgs;
    in
      python3Packages.buildPythonApplication {
        pname = "whispy";
        version = "0.1.0";
        format = "pyproject";
        nativeBuildInputs = [python3.pkgs.hatchling];
        src = ./.;
        propagatedBuildInputs = with python3Packages; [
          torchWithCuda
          transformers
          numpy
          pysilero-vad
          librosa
          pyaudio
        ];
        postInstall = ''
          mkdir -p $out/bin
          cp src/whispy.py $out/bin/whispy
          chmod +x $out/bin/whispy
        '';
      };
    mkVoiceInput = system: let
      pkgs = configuredNixpkgs.${system};
      voiceInputPkg = mkPackage system;
    in
      pkgs.stdenv.mkDerivation {
        name = "whispy";
        src = ./.;
        phases = ["unpackPhase" "installPhase"];
        buildInputs = with pkgs; [makeWrapper pipewire wtype playerctl voiceInputPkg];
        installPhase = ''
          mkdir -p $out/bin
          mkdir -p $out/share/whispy
          cp -r sounds $out/share/whispy/
          ln -s ${voiceInputPkg}/bin/whispy $out/bin/whispy
          wrapProgram $out/bin/whispy \
            --set SOUNDS_DIR $out/share/whispy/sounds
        '';
      };
  in {
    packages = forAllSystems (system: {
      default = mkVoiceInput system;
      whispy = mkPackage system;
    });
  };
}
