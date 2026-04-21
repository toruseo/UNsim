import os
import sys
import tempfile
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd().resolve()
    sys.path = [
        p for p in sys.path
        if (Path(p).resolve() if p else cwd) != repo_root
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        import unsim
        from unsim import World

        unsim_path = Path(unsim.__file__).resolve()
        if unsim_path.is_relative_to(repo_root):
            raise AssertionError(f"unsim imported from repository path: {unsim_path}")

        world = World(name="smoke", tmax=1200, print_mode=0, save_mode=0, show_mode=0)
        world.addNode("orig", x=0, y=0)
        world.addNode("dest", x=1, y=0)
        world.addLink("link", "orig", "dest", length=1000, free_flow_speed=20)
        world.adddemand("orig", "dest", t_start=0, t_end=1000, flow=0.3)
        world.exec_simulation()

        assert world.get_link("link").cum_departure[-1] > 0
        print("pip install smoke test passed")


if __name__ == "__main__":
    main()
