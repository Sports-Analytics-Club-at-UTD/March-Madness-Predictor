import pandas as pd
def main():
    print("Hello from backend!")
    df = pd.read_csv('test.csv')
    print(df)

if __name__ == "__main__":
    main()
