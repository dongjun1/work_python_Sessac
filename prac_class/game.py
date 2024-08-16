import random

class Game:
    def __init__(self, players):
        self.players = players
    
    def check_win_lose(self, player1, player2) :
        if player1.actual_rating > player2.actual_rating :
            return 1
        elif player1.actual_rating < player2.actual_rating :
            return 2
        else :
            return 'tie'
    
    def player_rate(self, player1, player2) :
        K = 1500
        rating_dev_1 = (player2.actual_rating - player1.actual_rating) / 400
        rating_dev_2 = (player1.actual_rating - player2.actual_rating) / 400
        We1 = 1 / (10 ** rating_dev_1 + 1)
        We2 = 1 / (10 ** rating_dev_2 + 1)
        
        if self.check_win_lose(player1,player2) == 1:
            player1.win_lose_history.append('win')
            player2.win_lose_history.append('lose')
            W1 = 1
            W2 = 0
            player1.current_rating += K * (W1 - We1)
            player2.current_rating += K * (W2 - We2)
            print(f'{player1,player_id} was win')
        elif self.check_win_lose(player1,player2) == 2:
            player1.win_lose_history.append('lose')
            player2.win_lose_history.append('win')
            W1 = 0
            W2 = 1
            player1.current_rating += K * (W1 - We1)
            player2.current_rating += K * (W2 - We2)
            print(f'{player2.player_id} was win')
        else :
            player1.win_lose_history.append('tie')
            player2.win_lose_history.append('tie')
            W1 = 0.5
            W2 = 0.5
            player1.current_rating += K * (W1 - We1)
            player2.current_rating += K * (W2 - We2)
            print('tie')

        # return player1.current_rating, player2.current_rating
         
    
    def play_match(self, player1, player2):
        # player1, player2의 win_lose_history를 update하고 
        # elo rating 알고리즘에 따라 각자의 current_rating을 update할 것 
        # https://namu.wiki/w/Elo%20%EB%A0%88%EC%9D%B4%ED%8C%85 참고 
        

        self.player_rate(player1, player2)
        return f"{player1.player_id}'s current_rating : {player1.current_rating}, {player2.player_id}'s current_rating : {player2.current_rating}"
        
            

    def match_players(self):
        
        # n = len(self.players) - 1
        # player1 = self.players[random.randint(0, n)]
        # player2 = self.players[random.randint(0, n)]
        
        # player들을 current_rating을 기반으로
        
        current_rating_lst = []

        for k, v in self.players.items() :
            current_rating_lst.append((k, v.current_rating))

        player1 = self.players[current_rating_lst[0][0]]
        player2 = self.players[current_rating_lst[1][0]]

    

        print(f"{player1.player_id}'s actual_rating : {player1.actual_rating}, {player2.player_id}'s actual_rating : {player2.actual_rating}")
        return self.play_match(player1, player2)

    def simulate(self):
        pass 

class Player:
    def __init__(self, player_id, initial_rating = 1000, actual_rating = 1000):
        self.win_lose_history = []
        self.current_rating = initial_rating
        self.actual_rating = actual_rating
        self.player_id = player_id
        

    def __str__(self):
        return str(self.player_id, self.current_rating)

    
# if __name__ == 'main' :

# player를 랜덤으로 생성해서 기능이 동작하는지 체크용
# player_list = []

# for i in range(10) :
#     
#     rand_player = Player(i, actual_rating = rand_ar)
#     player_list.append(rand_player)

# game = Game(player_list)
# res = game.match_players()
# print(res)

player_list = {}
dongjun = Player('dongjun', actual_rating = 2879)
dongjun = Player('dongjun', actual_rating = 500)
unknown = Player('unknown', actual_rating = 3054)
unknown = Player('unknown', actual_rating = 1500)
unknown2 = Player('unknown2', actual_rating = 4832)
unknown3 = Player('unknown3', actual_rating = 2150)
player_list['dongjun'] = dongjun
player_list['unknown'] = unknown
player_list['unknown3'] = unknown
player_list['unknown4'] = unknown

game = Game(player_list)
for i in range(100):
    res = game.match_players()

    print(res)
