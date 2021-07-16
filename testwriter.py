if __name__ == '__main__':
    main()

def main():
    with open('/test-logs/testlog.txt', 'w') as f:
        for i in range(len(100)):
            f.write(f'log: {i}')