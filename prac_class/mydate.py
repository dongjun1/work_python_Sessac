class MyDate:

    def __init__(self, year = 0, month = 0, day = 0, hour = 0, minute = 0, sec = 0):
        
        def leap_year(year) :
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                return True
            return False
        
        self.day_31 = [1, 3, 4, 7, 8, 10, 12]
        self.day_30 = [4, 6, 9, 11]
        
        is_leap_year = leap_year(year)


        def day_limit(month, day, is_leap_year) :
            
            day_limit = 0

            if month in self.day_31 :               
                day_limit = 31
            elif month == 0 :
                day_limit = day            
            else :
                day_limit = 30
                if month == 2 :  
                    day_limit = 28            
                    if is_leap_year :
                        day_limit = 29                                           
                        
            return day_limit

        self.day_limit = day_limit(month, day, is_leap_year)

        def check_valid(month, day, hour, minute, sec, day_31, is_leap_year, day_limit) :
            assert 0 <= month <= 12 , False
            assert 0 <= day <= day_limit, False
            assert 0 <= hour <= 24, False
            assert 0 <= minute <= 60, False
            assert 0 <= sec <= 60, False
            
            return True
        
        if check_valid(month, day, hour, minute, sec, self.day_31, is_leap_year, self.day_limit) :

            self.year = year
            self.month = month
            self.day = day
            self.hour = hour
            self.minute = minute
            self.sec = sec 
            # self.attr_list = (year, month, day, hour, minute, sec)
    
    def __add__(self, other):
        if isinstance(self, MyDate) and isinstance(other, MyDate) :
            add_year = self.year + other.year
            add_month = self.month + other.month
            add_day = self.day + other.day
            add_hour = self.hour + other.hour
            add_minute = self.minute + other.minute
            add_sec = self.sec + other.sec
            day_limit = self.day_limit

            if add_month > 12 :
                add_year += add_month // 12
                add_month = add_month % 12
            
            # day는 추후에 수정.
            # if add_month - 1 in self.day_31 :
            #     day_limit = self.day_limit
            # elif add_month - 1 in self.day_30 :
            #     day_limit = self.day_limit
            # elif add_month - 1 == 2 :
            #     day_limit = self.day_limit
            #     if ((add_year % 4) == 0 and (add_year % 100 != 0)) or (add_year % 400 == 0) :
            #         day_limit = self.day_limit
            if add_day > day_limit :
                add_month += add_day // day_limit
                
                if add_day // day_limit >= 2 :
                    
                    while add_day > day_limit :
                        add_day -= day_limit
                    add_day = add_day
                
            
            if add_hour >= 24 :
                add_day += add_hour // 24
                add_hour = add_hour % 24
            if add_minute >= 60 :
                add_hour += add_minute // 60
                add_minute = add_minute % 60
            if add_sec >= 60 :
                add_minute += add_sec // 60
                add_sec = add_sec % 60

            return MyDate(add_year, add_month, add_day, add_hour, add_minute, add_sec)
        else :
            raise TypeError       
    
    def __sub__(self, other):
        if isinstance(self, MyDate) and isinstance(other, MyDate) :
            sub_year = 0
            sub_month = 0
            sub_day = 0
            sub_hour = 0
            sub_minute = 0
            sub_sec = 0
            
            if not self == other :
                if self > other :
                    if self.sec < other.sec :
                        self.minute -= 1
                        self.sec += 60
                    sub_sec = self.sec - other.sec
                    if self.minute < other.minute :
                        self.hour -= 1
                        self.minute += 60
                    sub_minute = self.minute - other.minute
                    if self.hour < other.hour :
                        self.day -= 1
                        self.hour += 24
                    sub_hour = self.hour - other.hour
                    if self.day < other.day :
                        if self.month in self.day_31 :
                            self.month -= 1
                            self.day += 31
                        if self.month in self.day_30 :
                            self.month -= 1
                            self.day += 30
                        if self.month == 2 :
                            self.month -= 1
                            self.day += 28
                            if ((self.year % 4) == 0 and (self.year % 100 != 0)) or (self.year % 400 == 0) :
                                self.day += 29
                    sub_day = self.day - other.day
                    if self.month < other.month :
                        self.year -= 1
                        self.month += 12
                    sub_month = self.month - other.month

                    sub_year = self.year - other.year
                    
                    return MyDate(sub_year, sub_month, sub_day, sub_hour, sub_minute, sub_sec)
                else :
                    if self < other :
                        if other.sec < self.sec :
                            other.minute -= 1
                            other.sec += 60
                        sub_sec = other.sec - self.sec
                        if other.minute < self.minute :
                            other.hour -= 1
                            other.minute += 60
                        sub_minute = other.minute - self.minute
                        if other.hour < self.hour :
                            other.day -= 1
                            other.hour += 24
                        sub_hour = other.hour - self.hour
                        if other.day < self.day :
                            if other.month in other.day_31 :
                                other.month -= 1
                                other.day += 31
                            if other.month in other.day_30 :
                                other.month -= 1
                                other.day += 30
                            if other.month == 2 :
                                other.month -= 1
                                other.day += 28
                                if ((other.year % 4) == 0 and (other.year % 100 != 0)) or (other.year % 400 == 0) :
                                    other.day += 29
                        sub_day = other.day - self.day
                        if other.month < self.month :
                            other.year -= 1
                            other.month += 12
                        sub_month = other.month - self.month

                        sub_year = other.year - self.year
                        
                    return MyDate(sub_year, sub_month, sub_day, sub_hour, sub_minute, sub_sec)
        else :
            raise TypeError 

    def __eq__(self, other):
        if isinstance(self, MyDate) and isinstance(other, MyDate) :
            def check_eq(self, other):
                if (self.year == other.year and
                 self.month == other.month and
                 self.day == other.day and
                 self.hour == other.hour and
                 self.minute == other.minute and 
                 self.sec == other.sec) : return True

                else :
                    return False              
            
            return check_eq(self, other)
        else :
            raise TypeError

    # 미만
    def __lt__(self, other):
        if isinstance(self, MyDate) and isinstance(other, MyDate) :
            if not self == other :
                def check_lt(self, other) :
                    if self.year < other.year :
                        return True
                    else :
                        if self.year == other.year :
                            if self.month < other.month :
                                return True
                            else :
                                if self.month == other.month :
                                    if self.day < other.day :
                                        return True
                                    else :
                                        if self.day == other.day :
                                            if self.hour < other.hour :
                                                return True
                                            else :
                                                if self.hour == other.hour :
                                                    if self.minute < other.minute :
                                                        return True
                                                    else :
                                                        if self.minute == other.minute :
                                                            if self.sec < other.sec :
                                                                return True
                        
                        return False                
                    
                    # return all([a==b for a, b in zip(self.attr_list, other.attr_list)])
                    
                return check_lt(self, other)
        else :
            raise TypeError 
    
    # 이하
    def __le__(self, other):
        if isinstance(self, MyDate) and isinstance(other, MyDate) :
            if self == other or self < other :
                return True
            else :
                return False
        else :
            raise TypeError 

    # 초과
    def __gt__(self, other):
        if isinstance(self, MyDate) and isinstance(other, MyDate) :
            if not self < other :
                return True
            else :
                return False
        else :
            raise TypeError 

    # 이상
    def __ge__(self, other):
        if isinstance(self, MyDate) and isinstance(other, MyDate) :
            if self == other or not self < other :
                return True
            else :
                return False
        else :
            raise TypeError

    def __str__(self)  :
        return str((self.year, self.month, self.day, self.hour, self.minute, self.sec))

if __name__ == '__main__':
    # d0 = MyDate()
    d1 = MyDate(2022, 4, 1, 14, 30)
    d2 = MyDate(2024, 4, 1, 14, 30)
    # d2 = MyDate(2024, 8, 100, 23, 10) # should raise an error 
    # d3 = MyDate(2024, 2, 30) # should raise an error
    # d3 = MyDate(day = 1)
    # print(d3.month, d3.day)
    # res = d1 + d3
    # assert d1 + d3 == MyDate(2022, 4, 2, 14, 30)
    # print(d1 + d3 != MyDate(2022, 4, 2, 14, 30))
    # assert d1 - d3 == MyDate(2022, 3, 31, 14, 30) 
    d3 = MyDate(2022, 1, 31)
    d4 = MyDate(day = 31)

    print(d3 + d4)


    
    
