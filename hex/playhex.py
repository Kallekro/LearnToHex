import sys, os, subprocess, select, atexit, time, argparse
import tkinter as tk

def closePipe(pipe):
    try:
        pipe.flush()
        pipe.close()
    except: # already closed
        pass

def closeHex(hex_process):
    closePipe(hex_process.stdin)
    closePipe(hex_process.stdout)
    closePipe(hex_process.stderr)
    # kill the process
    try:
        hex_process.kill()
    except:
        pass

def startHex():
    hex_process = subprocess.Popen(['build/hex_python'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atexit.register(lambda: closeHex(hex_process))
    return hex_process

def readAllDecorator(fun):
    def readAll(self):
        while fun(self): continue
    return readAll

class HexInterface():
    def __init__(self):
        self.hex_process = startHex()
        self.board_size = 0
        self.recent_board = ""
        self.reading_board = False
        self.game_running = True
        self.player_won = -1

    def restart(self):
        self.recent_board = ""
        self.reading_board = False
        self.game_running = True
        self.player_won = -1
        closeHex(self.hex_process)
        self.hex_process = startHex()


    def sendInput(self, inp):
        if not self.game_running: return
        rlist, wlist, xlist = select.select([], [self.hex_process.stdin], [])
        os.write(wlist[0].fileno(), (inp.strip('\n') + '\n').encode(sys.stdin.encoding))
        self.readOutput()

    @readAllDecorator
    def readOutput(self):
        if not self.game_running: return
        rlist, wlist, xlist = select.select([self.hex_process.stdout], [], [])
        output = ""
        for stdout in rlist:
            output += os.read(stdout.fileno(), 1024).decode(sys.stdout.encoding)
        for line in output.split('\n'):
            if line.startswith("__BOARD_SIZE__"):
                self.board_size = int(line.split(' ')[1])
            elif line.startswith("__GAME_OVER__"):
                self.game_running = False
                self.player_won = int(line.split()[1])
                return 0
            elif line == "__BOARD_BEGIN__":
                self.recent_board = ""
                self.reading_board = True
            elif line == "__BOARD_END__":
                self.reading_board = False
            elif line == "__TURN__":
                return 0
            elif line == "__INVALID_INPUT__":
                sys.stdout.write("Invalid input, please try again..\n")
                return 0
            elif line == "__NON_EMPTY__":
                return 0
            elif len(line):
                if self.reading_board:
                    self.recent_board += line + '\n'
                else:
                    sys.stdout.write(line + '\n')
        return 1

class HexGUI(tk.Canvas):
    def __init__(self, master, board_size, *args, **kwargs):
        self.board_size = board_size
        self.tile_line_coeff = 10
        self.tile_size = 10 * 4
        self.tiles_start_pos = (30, 30)
        self.board = [[None for _ in range(board_size)] for _ in range(board_size)]
        tk.Canvas.__init__(self, master,  width=board_size * self.tile_size * 1.5, height=board_size * self.tile_size, *args, **kwargs)

        #self.bind("<Button-1>", self.master.clicked)


    def drawTile(self, i, j, tile_state):
        x = j * self.tile_size + self.tiles_start_pos[0] + i * self.tile_size / 2
        y = i * self.tile_size + self.tiles_start_pos[1] - i * self.tile_line_coeff
        if tile_state == 0:
            color = 'blue'
        elif tile_state == 1:
            color = 'red'
        else:
            color = 'gray'

        tile = self.create_polygon(
            [
                x, y,
                x + 2*self.tile_line_coeff, y+self.tile_line_coeff,
                x+self.tile_line_coeff*2, y+self.tile_line_coeff*3,
                x, y+self.tile_line_coeff*4,
                x-self.tile_line_coeff*2, y+self.tile_line_coeff*3,
                x-self.tile_line_coeff*2, y+self.tile_line_coeff
            ],
            outline='black', fill=color)

        self.tag_bind(tile, '<Button-1>', lambda _: self.master.clicked(i, j))

class HexApp(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid(padx=5, pady=5)

        self.hex_interface = HexInterface()
        self.hex_interface.readOutput()

        self.createWidgets()

        self.drawBoard()

    def createWidgets(self):
        if self.hex_interface.board_size == 0:
            sys.exit("Must read beginning output from hex before creating widgets.")

        restart_button = tk.Button(self, text="Restart", command=self.reset)
        restart_button.grid()

        self.hex_GUI = HexGUI(self, self.hex_interface.board_size, background='white', bd=3, relief=tk.SUNKEN)
        self.hex_GUI.grid()

        self.win_stringvar = tk.StringVar()
        self.win_stringvar.set("")
        win_label = tk.Label(self, textvariable=self.win_stringvar)
        win_label.grid()

    def drawBoard(self):
        self.hex_GUI.delete(tk.ALL)
        board = self.hex_interface.recent_board.split('\n')
        for i in range(len(board)):
            for j in range(len(board[i])):
                self.hex_GUI.drawTile(j,i, int(board[j][i]))

    def clicked(self, i, j):
        hexinput = chr(ord('A') + j) + str(i+1)
        self.hex_interface.sendInput(hexinput)
        self.drawBoard()
        if not self.hex_interface.game_running:
            if self.hex_interface.player_won == 0:
                self.win_stringvar.set("Blue won!")
            elif self.hex_interface.player_won == 1:
                self.win_stringvar.set("Red won!")

    def reset(self):
        self.win_stringvar.set("")
        self.hex_interface.restart()
        self.hex_interface.readOutput()
        self.drawBoard()

def main():
    root = tk.Tk()
    app = HexApp(root)
    app.master.title("Hex")
    app.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play hex.")
    parser.add_argument("--make", dest="make", action='store_true')
    args = parser.parse_args()

    if args.make:
        if subprocess.run(["cd build && make hex_python && cd .."], shell=True).returncode != 0:
            sys.exit("Failed to make.")

    main()

#updateLoop(startHex())