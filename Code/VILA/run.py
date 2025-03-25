import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run VILA-13b evaluation scripts.")
    parser.add_argument('--model', type=str, required=True, choices=['VILA'],
                        help='Model to run (VILA)')
    parser.add_argument('--method', type=str, required=True, choices=['zeroshot', 'fewshot', 'COT', 'ICL'],
                        help='Method to use (zeroshot, fewshot, COT, ICL)')
    parser.add_argument('--step', type=str, required=True, choices=['Step2'],
                        help='Step to run (Step2)')

    args = parser.parse_args()

    script_path = f"{args.model}_{args.method}_{args.step}.py"
    print(f"Running: {script_path}")

    subprocess.run(['python', script_path])

if __name__ == "__main__":
    main()
