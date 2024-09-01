import os
import subprocess

def run_setup_script():
    # Path to the setup.sh script
    setup_script_path = os.path.join(os.getcwd(), 'setup.sh')

    # Make the script executable
    os.chmod(setup_script_path, 0o755)

    # Run the script
    subprocess.call(setup_script_path, shell=True)

if __name__ == "__main__":
    run_setup_script()
