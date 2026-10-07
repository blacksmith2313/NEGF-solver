import re
import subprocess
import sys

# device profiles script
PROF = "device_profile.py"
SOLVER_SCRIPT = "poisson_solver.py"

def run_solver():

    subprocess.run([sys.executable, SOLVER_SCRIPT], check=True)

    with open(PROF, "r") as f:
        original_content = f.read()

    pattern = re.compile(r'mode\s*=\s*"phonon"')

    new_pattern = 'mode = "ballistic"' 

    updated_code = pattern.sub(new_pattern, original_content)

    with open(PROF, "w") as f:
        f.write(updated_code)

    subprocess.run([sys.executable, SOLVER_SCRIPT], check=True)

    return


if __name__ == "__main__":

    run_solver()
