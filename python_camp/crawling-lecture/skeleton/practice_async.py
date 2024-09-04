import time
import asyncio

# def task(task_number, duration):
#     print(f'started {task_number}')
#     time.sleep(duration)
#     print(f'ended {task_number}')

# def main():
#     begin = time.time()
#     task1 = (1, 3)
#     task2 = (2, 2)
#     task3 = (3, 1)

#     task(*task1)
#     task(*task2)
#     task(*task3)
#     end = time.time()

#     print(end - begin)

# main()

# 비동기 프로그래밍 : 팀플하는 것과 같음. 일반적으로 사람끼리 협업을 할 경우 각자 역할 분담을 하고 각자의 역할에 맞는 업무를 각자 동시에 수행한 뒤 취합.
#                  하지만 프로그래밍은 그렇지 않음. 위에서 실행시킨 코드가 실행되어 계산이 완료될 때까지 하위의 코드는 실행되지 않음.
#                  이러한 것을 사람이 하는 것처럼 시간이 소요되는 코드를 동시에 실행시켜 코드의 실행시간을 단축할 수 있음.
# interupt handling
async def async_task(task_number, duration):
    print(f'started {task_number}')
    await asyncio.sleep(duration)
    print(f'ended {task_number}')

async def main():
    task1 = asyncio.create_task(async_task(1, 3))
    task2 = asyncio.create_task(async_task(2, 2))
    task3 = asyncio.create_task(async_task(3, 1))

    await task1
    await task2
    await task3

async def gather_main():
    await asyncio.gather(async_task(1, 3), async_task(2, 2), async_task(3, 1))

begin = time.time()
# asyncio.run(main())
asyncio.run(gather_main())
end = time.time()

print(end - begin)