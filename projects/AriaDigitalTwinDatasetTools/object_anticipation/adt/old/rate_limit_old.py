import threading
import time 
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for rate limiting
lock = threading.Lock()
requests_made = 0
start_time = time.time()

def rate_limiter():
    global requests_made, start_time
    with lock:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > 60:
            requests_made = 0
            start_time = current_time
        if requests_made >= 500:
            sleep_time = 60 - elapsed_time
            logger.debug(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
            print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
            requests_made = 0
            start_time = time.time()
        requests_made += 1
        logger.debug(f"Requests made: {requests_made}")

import asyncio

# Global variables for rate limiting
alock = asyncio.Lock()
requests_made = 0
start_time = time.time()

async def async_rate_limiter():
    global requests_made, start_time
    async with alock:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > 60:
            # Reset every minute
            requests_made = 0
            start_time = current_time
        if requests_made >= 500:
            sleep_time = 60 - elapsed_time
            logger.debug(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
            print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
            await asyncio.sleep(sleep_time)
            # Reset after sleeping
            requests_made = 0
            start_time = time.time()
        requests_made += 1
        logger.debug(f"Requests made: {requests_made}")

class AsyncRateLimiter:
    def __init__(self, max_calls, period):
        self._max_calls = max_calls
        self._period = period
        self._calls = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            current = time.monotonic()
            # Remove calls that are outside the time window
            while self._calls and self._calls[0] <= current - self._period:
                self._calls.pop(0)
            if len(self._calls) >= self._max_calls:
                sleep_time = self._period - (current - self._calls[0])
                await asyncio.sleep(sleep_time)
            self._calls.append(time.monotonic())

rate_limiter = AsyncRateLimiter(max_calls=500, period=60)
