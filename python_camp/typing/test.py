def EvenOddChecker(number: int) -> str:
    if number % 2 == 0:
        return 'even'
    else:
        return 'odd'

if __name__ == '__main__':
    print(EvenOddChecker(5))