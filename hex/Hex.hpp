#ifndef HEX_HPP
#define HEX_HPP
#include <shark/LinAlg/Base.h>
#include <string>

namespace Hex {

    enum TileState : unsigned int {
        Blue = 0,
        Red = 1,
        Empty = 2
    };

    struct LineSegment {
        unsigned int m_count;
        bool m_connected_A = false;
        bool m_connected_B = false;

    public:
        std::pair<unsigned int, unsigned int>* TilePositions;
        unsigned Count() { return m_count; }
        unsigned ConnectedA() { return m_connected_A; }
        unsigned ConnectedB() { return m_connected_B; }
        LineSegment (std::pair<unsigned int, unsigned int>* tile_positions) {
            TilePositions = tile_positions;
            // tile_positions = new std::pair<unsigned int, unsigned int>[count]
        }
    };

    class Game {
        static const int BOARD_SIZE = 11;

	    shark::blas::matrix<TileState> m_gameboard;
        unsigned int m_current_player = 0;

        const std::string m_red_color = "\033[1;31m";
        const std::string m_blue_color = "\033[1;34m";
        const std::string m_reset_color = "\033[0;0m";
        const std::string m_hexes[3] = {
            m_blue_color + "\u2b22" + m_reset_color, // Blue
            m_red_color + "\u2b22" + m_reset_color,  // Red
            "\u2b21"                                 // Empty
        };

        void m_next_player() {
            m_current_player = (m_current_player + 1) % 2;
        }

        void m_initializeboard() {
            for (int i=0; i < BOARD_SIZE; i++) {
                for (int j=0; j < BOARD_SIZE; j++) {
                    m_gameboard(i,j) = Empty;
                }
            }
        }

        void m_printletters(int start_spaces) {
            for (int i = 0; i < start_spaces; i++) {
                std::cout << " ";
            }
            for (char i = 65; i < 65+BOARD_SIZE; i++) {
                std::cout << m_red_color << i << m_reset_color << " ";
            }
            std::cout << std::endl;
        }

        bool m_place_tile(std::pair<unsigned int, unsigned int> pos) {
            m_gameboard(pos.first, pos.second) = (TileState)m_current_player;
        }

        std::pair<unsigned int,unsigned int> m_alphnum2num(std::string position) {
            return std::pair<int,int>(atoi(position.substr(1, position.length()-1).c_str()) - 1, toupper(position[0]) - 65);
        }

    public:
        Game() : m_gameboard(BOARD_SIZE, BOARD_SIZE) {
            m_initializeboard();
        }

        void take_turn(std::string pos) {
            m_place_tile(m_alphnum2num(pos));
            m_next_player();
        }

        void printhex( ) {
            m_printletters(1);
            for (int i=0; i < BOARD_SIZE; i++) {
                for (int ii=0; ii < i; ii++) {
                    if (ii == 8) { continue; }
                    std::cout << " ";
                }
                std::cout << m_blue_color << i+1 << m_reset_color;
                for (int j=0; j < BOARD_SIZE; j++) {
                    std::cout << " " << m_hexes[m_gameboard(i, j)];
                }
                std::cout << " " << m_blue_color << i+1 << m_reset_color;
                std::cout << std::endl;
            }
            m_printletters(BOARD_SIZE + 2);
        }
    };
}

#endif