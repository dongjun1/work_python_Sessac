game_status = {'x_positions' : [], 'o_positions' : []} 

def empty_board(x_size = 3, y_size = 3, x_cell_size = 5, y_cell_size = 3):
    """Create an empty board. 

    The board is made of horizontal lines, made with - and vertical lines, made with |. 

    (optional) When there are no x_cell_size and y_cell_size arguments, default to arbitary size of your choice. Just make it consistent. 
    """
    h_line = ('  ' + '-' * x_cell_size) * x_size
    v_line = '|' + ' ' * x_cell_size
    
    for x in range(y_size) :
        print(h_line)
        for z in range(y_cell_size) :
            for y in range(x_size) :
                print(v_line, end = ' ')
            print('|')
    print(h_line)


def play(game_status, x_or_o, coordinate):
    """Main function for simulating tictactoe game moves. 

    Tictactoe game is executed by two player's moves. In each move, each player chooses the coordinate to place their mark. It is impossible to place the mark on already taken position. 

    A move in the tictactoe game is composed of two components; whether who ('X' or 'O') made the move, and how the move is made - the coordinate of the move. 

    Coordinate in our tictactoe system will use the coordinate system illustrated in the example below. 
    
    Example 1. 3 * 4 tictactoe board. 
    
         ---------- ---------- ----------
        |          |          |          |
        |  (0,0)   |  (1,0)   |  (2,0)   |
        |          |          |          |
         ---------- ---------- ----------
        |          |          |          |
        |  (0,1)   |  (1,1)   |  (2,1)   |
        |          |          |          |
         ---------- ---------- ----------
        |          |          |          |
        |  (0,2)   |  (1,2)   |  (2,2)   |
        |          |          |          |
         ---------- ---------- ----------
        |          |          |          |
        |  (0,3)   |  (1,3)   |  (2,3)   |
        |          |          |          |
         ---------- ---------- ----------
        """
    if x_or_o == 'X' :
        if coordinate not in game_status['x_positions'] + game_status['o_positions'] :
            game_status['x_positions'].append(coordinate) 
            
    elif x_or_o == 'O' :
        if coordinate not in game_status['x_positions'] + game_status['o_positions'] :
            game_status['o_positions'].append(coordinate)
            

def check_winlose(game_status):
    """Check the game status; game status should be one of 'X wins', 'O wins', 'tie', 'not decided'. 
    """
    list_x_x, list_x_y = [i[0] for i in game_status['x_positions']] , [i[1] for i in game_status['x_positions']]
    
    list_o_x, list_o_y = [i[0] for i in game_status['o_positions']] , [i[1] for i in game_status['o_positions']]

    # 조건 수정할 것! 1) x좌표가 연속적, y좌표가 고정 2) y좌표가 연속적, x좌표가 고정 3) x,y좌표 모두 연속적. 
    if list_x_x == [0, 1, 2] and list_x_y == [0, 1, 2] :
        return "X wins!"
    elif list_o_x == [0, 1, 2] and list_o_y == [0, 1, 2] :
        return "O wins!"
    elif len(list_x_x + list_o_x) == 9 :
        return "tie"
    else :
        return 'not decided'
    
    '''
    x_pos = game_status['x_positions']

    winning_positions = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]

    if determine_if_x_wins(game_status, winning_positions) :
        return 'X wins'
    elif determine_if_o_wins(game_status, winning_positions) :
        return 'O wins'
    else :
        if len(game_status['x_positions'] + game_status['o_positions']) == 9 :
            return 'Tie'
        else :
            return 'Not decided'


    def determine_if_x_wins(game_status, winning_positions) :
        for win in winning_positions:
            a, b, c = win
            if a in x_pos and b in x_pos and c in x_pos :
                return True
        return False
    
    def determine_if_o_wins(game_status, winning_positions) :
        for win in winning_positions:
            a, b, c = win
            if a in o_pos and b in o_pos and c in o_pos :
                return True
        return False
    '''
def display(game_status, x_size = 3, y_size = 3, x_cell_size = 10, y_cell_size = 7):
    """Display the current snapshot of the board. 

    'Snapshot' should contain following components. 

    - The board itself 
    - Moves that are already made

    For clarification, see provided examples. 

    Example 1. 
    When TictactoeGame instance t have following attributes; 
    - x_positions = [(0,0), (2,0), (2,1), (1,2)]
    - o_positions = [(1,0), (1,1), (0,2), (2,2)]

    t.display()
    >> 
     ---------- ---------- ----------
    |          |          |          |
    |    X     |    O     |    X     |
    |          |          |          |
     ---------- ---------- ----------
    |          |          |          |
    |          |    O     |    X     |
    |          |          |          |
     ---------- ---------- ----------
    |          |          |          |
    |    O     |    X     |    O     |
    |          |          |          |
     ---------- ---------- ----------

    """

    h_line = ('  ' + '-' * x_cell_size) * x_size
    v_line = '|' + ' ' * x_cell_size
    
    for x in range(y_size) :
        print(h_line)
        for z in range(y_cell_size) :
            for y in range(x_size) :
                if z == 1 :
                    if (x, y) in game_status['x_positions'] :
                        print('|' + ' ' * (x_cell_size - 1) + 'X', end = ' ')
                    elif (x, y) in game_status['o_positions'] :
                        print('|' + ' ' * (x_cell_size - 1) + 'O', end = ' ')
                    else :
                        print('|' + ' ' * (x_cell_size), end = ' ')
                else :
                    print('|' + ' ' * (x_cell_size), end = ' ')
            print('|')
    print(h_line)

    

if __name__ == '__main__' :

    #empty_board(x_size = 3, y_size = 3, x_cell_size = 10, y_cell_size = 7)
    game_status = {'x_positions' : [(0,0)], 'o_positions' : [(1,0)]}
    display(game_status)