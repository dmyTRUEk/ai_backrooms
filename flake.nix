# src: https://tonyfinn.com/blog/nix-from-first-principles-flake-edition/nix-8-flakes-and-developer-environments
{
	inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

	outputs = { self, nixpkgs }:
	let
		system = "x86_64-linux";
		pkgs = import nixpkgs {
			inherit system;
			config.allowUnfree = true;
		};
	in {
		devShells.${system}.default = pkgs.mkShell {
			packages = with pkgs; [
				(pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
					torch
					# torchWithRocm
					# torchWithVulkan
					# torchvision
					# torchvision-bin
					pillow
					numpy
				]))
				# Add other dependencies here:
			];
			# Set environment variables here:
			# MY_ENV_VAR = 1;
		};
		# Define extra shells or packages here.
	};
}
