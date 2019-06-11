# This program is a tk GUI application that can be used to play Hex against a trained strategy
# The program relies on the target 'hex_python' which is built with make. Running this file with --make, the target is built.
# The program works by opening the hex C++ program as a subprocess and communicating with it following a strict protocol.
import sys, os, subprocess, select, atexit, time, argparse
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

""" Close a pipe if open. """
def closePipe(pipe):
    try:
        pipe.flush()
        pipe.close()
    except: # already closed
        pass

""" Close the hex process if alive """
def closeHex(hex_process):
    closePipe(hex_process.stdin)
    closePipe(hex_process.stdout)
    closePipe(hex_process.stderr)
    # kill the process
    try:
        hex_process.kill()
    except:
        pass

""" Start the hex process """
def startHex(model="", what=""):
    hex_process = subprocess.Popen(['build/hex', what, model], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atexit.register(lambda: closeHex(hex_process))
    return hex_process

""" The interface between the hex subprocess and the main process. """
class HexInterface():
    def __init__(self, model="", what=""):
        self.model = model
        self.what = what
        self.startHexSafe()
        self.board_size = 0
        self.recent_board = ""
        self.reading_board = False
        self.game_running = True
        self.player_won = -1

    def startHexSafe(self):
        self.hex_process = startHex(self.model, self.what)
        rlist, wlist, xlist = select.select([self.hex_process.stdout], [], [])
        while True:
            output = ""
            for stdout in rlist:
                output += os.read(stdout.fileno(), 1024).decode(sys.stdout.encoding)
            for line in output.split('\n'):
                if line == "__MODEL_GOOD__":
                    rlist, wlist, xlist = select.select([], [self.hex_process.stdin], [])
                    os.write(wlist[0].fileno(), '__BEGIN__\n'.encode(sys.stdin.encoding))
                    return
                elif line == "__MODEL_BAD__":
                    self.model = ""
                    messagebox.showerror("Model error", "The model could not be loaded. Are the dimensions correct?")
                    self.hex_process = startHex()
                    rlist, wlist, xlist = select.select([], [self.hex_process.stdin], [])
                    os.write(wlist[0].fileno(), '__BEGIN__\n'.encode(sys.stdin.encoding))
                    return

    def restart(self):
        self.recent_board = ""
        self.reading_board = False
        self.game_running = True
        self.player_won = -1
        closeHex(self.hex_process)
        self.startHexSafe()

    def sendInput(self, inp):
        if not self.game_running: return
        rlist, wlist, xlist = select.select([], [self.hex_process.stdin], [])
        os.write(wlist[0].fileno(), (inp.strip('\n') + '\n').encode(sys.stdin.encoding))
        self.readOutput()

    def readOutput(self):
        if not self.game_running: return
        rlist, wlist, xlist = select.select([self.hex_process.stdout], [], [])
        read_prompt = False
        while not read_prompt:
            output = ""
            for stdout in rlist:
                output += os.read(stdout.fileno(), 1024).decode(sys.stdout.encoding)
            for line in output.split('\n'):
                if line.startswith("__BOARD_SIZE__"):
                    self.board_size = int(line.split(' ')[1])
                elif line.startswith("__GAME_OVER__"):
                    self.game_running = False
                    self.player_won = int(line.split()[1])
                    read_prompt = True
                elif line == "__BOARD_BEGIN__":
                    self.recent_board = ""
                    self.reading_board = True
                elif line == "__BOARD_END__":
                    self.reading_board = False
                elif line in ["__TURN__", "__INVALID_INPUT__", "__NON_EMPTY__"]:
                    read_prompt = True
                elif line == "__MODEL_GOOD__":
                    pass
                elif len(line):
                    if self.reading_board:
                        self.recent_board += line + '\n'
                    else:
                        sys.stdout.write(line + '\n')

""" The hex GUI is a canvas that shows the tiles. """
class HexGUI(tk.Canvas):
    def __init__(self, master, board_size, *args, **kwargs):
        self.board_size = board_size
        self.tile_line_coeff = 10
        self.tile_size = self.tile_line_coeff * 4
        self.tiles_start_pos = (self.tile_line_coeff * 3, self.tile_line_coeff)
        canvas_width = board_size * self.tile_size + board_size * 0.5 * self.tile_size
        canvas_height = board_size * self.tile_size - (board_size - 2) * self.tile_line_coeff
        tk.Canvas.__init__(self, master,  width=canvas_width, height=canvas_height, *args, **kwargs)

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

""" The tk application """
class HexApp(tk.Frame):
    def __init__(self, master=None, model="", what=""):
        tk.Frame.__init__(self, master)
        self.grid(padx=5, pady=5)

        self.hex_interface = HexInterface(model, what)
        self.hex_interface.readOutput()

        self.createWidgets()

        self.drawBoard()

    def createWidgets(self):
        if self.hex_interface.board_size == 0:
            sys.exit("Must read beginning output from hex before creating widgets.")

        modelframe = tk.Frame(self)
        modelframe.grid(row=0, column=0, sticky="W", padx=(0, 5))
        tk.Label(modelframe, text="Model:").grid(row=0, column=1, columnspan=2, sticky="W")
        loadmodel_button = tk.Button(modelframe, text= u"\u1A00", command=self.loadModel,padx=0)
        loadmodel_button.grid(row=1, column=0, sticky="W")
        self.model_stringvar = tk.StringVar()
        self.updateModelStringvar()
        tk.Label(modelframe, textvariable=self.model_stringvar).grid(row=1, column=1, sticky="W")

        restart_button = tk.Button(self, text="Restart", command=self.reset)
        restart_button.grid(row=0, column=1)

        self.hex_GUI = HexGUI(self, self.hex_interface.board_size, background='white', bd=3, relief=tk.SUNKEN)
        self.hex_GUI.grid(row=1, column=0, columnspan=2)

        self.win_stringvar = tk.StringVar()
        self.win_stringvar.set("")
        win_label = tk.Label(self, textvariable=self.win_stringvar)
        win_label.grid(row=2, column=0, columnspan=2)

    def updateModelStringvar(self):
        if len(self.hex_interface.model):
            self.model_stringvar.set(os.path.basename(os.path.normpath(self.hex_interface.model)))
        else:
            self.model_stringvar.set("None")

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

    def loadModel(self):
        filename = filedialog.askopenfilename(initialdir=__file__, title="Select model", filetypes=(("model", "*.model"), ("all files", "*.*")))
        if filename:
            self.hex_interface.model = filename
            self.reset()
            self.updateModelStringvar()

def main(model, algorithm):
    root = tk.Tk()
    if algorithm.upper() != "TD" and algorithm.upper() != "ES":
        sys.exit("Algorithm must be TD or ES.")
    algorithm += "python"
    app = HexApp(root, model, algorithm)
    app.master.title("Hex")
    app.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play hex.")
    parser.add_argument("--model", dest="model", default="")
    parser.add_argument("--make", dest="make", action='store_true')
    parser.add_argument("--algorithm", dest="algorithm", required=True, help="es/td")
    args = parser.parse_args()

    if args.make:
        if subprocess.run(["cd build && make hex_python && cd .."], shell=True).returncode != 0:
            sys.exit("Failed to make.")

    main(args.model, args.algorithm)
