from random import randint
from tictactoe_without_class import check_winlose 

def random_player(x_or_o, x_positions, o_positions):
    move = (0, 0)
    while move in x_positions + o_positions:
        x = randint(0, 2)
        y = randint(0, 2)
        move = (x, y)
    return move 

def smart_player(x_or_o, x_positions, o_positions):
    move = (0, 0)
    x = 0
    y = 0
    while move in x_positions + o_positions:
            
        if x_or_o == 'X' :
            if x_positions + o_positions == [] :
                x = 1
                y = 1
                move = (x,y)
                
            elif len(x_positions + o_positions) == 1 :
                x = randint(0, 2)
                y = randint(0, 2)
                rmove = (x,y)
               
            else :
                x, y = o_positions[-1]
            
                if x >= 1 and x == 0 and x < 2:
                    x += 1
                elif x <= 2 and x != 0 :
                    x -= 1
                
                if y >= 1 and y == 0 and y < 2:
                    y += 1
                elif y <= 2 and y != 0 :
                    y -= 1
        
        elif x_or_o == 'O' :
            if x_positions + o_positions == [] :
                x = 1
                y = 1
                move = (x,y)
                return move
            elif len(x_positions + o_positions) == 1 :
                x = randint(0, 2)
                y = randint(0, 2)
                move = (x,y)
                return move
            else :
                x, y = x_positions[-1]
                
                if x >= 1 and x == 0 and x < 2 :
                    x += 1
                elif x <= 2 and x != 0 :
                    x -= 1
                
                if y >= 1 and y == 0 and y < 2 :
                    y += 1
                elif y <= 2 and y != 0 :
                    y -= 1
        move = (x, y)
    return move
        

    #return random_player(x_or_o, x_positions, o_poistions)