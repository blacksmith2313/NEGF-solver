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

    pattern1 = re.compile(r'mode\s*=\s*"phonon"')
    pattern2 = re.compile(r'mode\s*=\s*"ballistic"')

    if pattern2.search(original_content):
        new_pattern = 'mode = "phonon"' 
        updated_code = pattern2.sub(new_pattern, original_content)
    elif pattern1.search(original_content):
        new_pattern = 'mode = "ballistic"' 
        updated_code = pattern1.sub(new_pattern, original_content)

    with open(PROF, "w") as f:
        f.write(updated_code)

    subprocess.run([sys.executable, SOLVER_SCRIPT], check=True)

    return


if __name__ == "__main__":

    run_solver()
