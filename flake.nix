{
  description = "Photogrammetry dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      platform = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${platform};
    in
    {
      devShells.${platform}.default = pkgs.mkShell {
        name = "photogrammetry-dev-shell";

        buildInputs = with pkgs; [
          gcc
          clang-tools
          cmake
          clinfo
          ocl-icd
          vulkan-tools
          vulkan-headers
          vulkan-loader
          vulkan-memory-allocator
          vulkan-validation-layers
          shaderc
          gtest
          pkg-config
          libx11
          libxrandr
          libxinerama
          libxcursor
          libxi
          opencv
          python3
          eigen
        ];

        VULKAN_SDK = "${pkgs.vulkan-loader}";
        VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";

        LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath (
          with pkgs;
          [
            ocl-icd
            vulkan-loader
            libx11
            libxrandr
            libxinerama
            libxcursor
            libxi
            opencv
          ]
        )}";

        CMAKE_INCLUDE_PATH = "${pkgs.vulkan-memory-allocator}/include";

        CMAKE_EXPORT_COMPILE_COMMANDS = "ON";

        shellHook = ''
          export SHELL="${pkgs.bashInteractive}/bin/bash"
          zeditor .
        '';
      };
    };
}
