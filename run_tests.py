import subprocess, os

def main():
    test_dir = "leo_segmentation/tests"
    
    for test_name in os.listdir(test_dir):
        if "test" in test_name:
            subprocess.call(["python","-m", "unittest", f"{test_dir}/{test_name}"], env=dict(os.environ), cwd="./")

if __name__ == "__main__":
    main()