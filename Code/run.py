import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run model evaluation scripts.")
    parser.add_argument('--model', type=str, required=True, choices=['Claude', 'Flash', 'GPT', 'GPTmini', 'Pro', 'VILA'],
                        help='Model to run (Claude, Flash, GPT, GPTmini, Pro, VILA)')
    parser.add_argument('--method', type=str, required=True, choices=['zeroshot', 'fewshot', 'COT'],
                        help='Method to use (zeroshot, fewshot, COT)')
    parser.add_argument('--step', type=str, required=True, choices=['Step1', 'Step2'],
                        help='Step to run (Step1 or Step2)')

    args = parser.parse_args()

    script_path = f"{args.method}/{args.model}_{args.method}_{args.step}.py"
    print(f"Running: {script_path}")

    subprocess.run(['python', script_path])

if __name__ == "__main__":
    main()
