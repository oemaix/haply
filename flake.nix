{
  description = "Haply development and test environment";

  inputs = {
    # Tarball pin, not github:: unauthenticated github: fetches hit the API
    # rate limit. Same release track as the maintainer hosts (nixos-26.05).
    nixpkgs.url = "https://github.com/NixOS/nixpkgs/archive/refs/heads/nixos-26.05.tar.gz";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      lib = nixpkgs.lib;

      mkPkgs =
        { unfreeDyalog ? false }:
        import nixpkgs {
          inherit system;
          config = lib.optionalAttrs unfreeDyalog {
            allowUnfreePredicate = pkg: lib.getName pkg == "dyalog";
            dyalog.acceptLicense = true;
          };
        };

      pythonPackages =
        ps: with ps; [
          hy
          numpy
          pytest
          torch
          uv
        ];

      mkShell =
        pkgs:
        pkgs.mkShell {
          packages = [ (pkgs.python3.withPackages pythonPackages) ];
          # Official loop is PYTHONPATH, not an editable install (decision 11).
          env.PYTHONPATH = toString ./.;
        };

      pkgs = mkPkgs { };
      oraclePkgs = mkPkgs { unfreeDyalog = true; };
    in
    {
      devShells.${system} = {
        # Official path (decision 11): Python, Hy, PyTorch, NumPy, pytest, uv.
        default = mkShell pkgs;

        # Optional Dyalog for a later oracle (decision 54). Unfree.
        oracle = oraclePkgs.mkShell {
          packages = [
            (oraclePkgs.python3.withPackages pythonPackages)
            oraclePkgs.dyalog
          ];
        };
      };
    };
}
