#ifndef HEX_HPP
#define HEX_HPP
#include <shark/LinAlg/Base.h>
#include <string>
#include <memory>

namespace Hex {

    enum TileState : unsigned {
        Blue = 0,
        Red = 1,
        Empty = 2
    };

    class LineSegment {
    public:
        bool Connected_A = false;
        bool Connected_B = false;
        LineSegment (bool _connected_A, bool _connected_B) {
            Connected_A = _connected_A;
            Connected_B = _connected_B;
        }

        bool Merge(std::shared_ptr<LineSegment> other) {
            Merge(other->Connected_A, other->Connected_B);
        }

        bool Merge(bool other_A, bool other_B) {
            Connected_A |= other_A;
            Connected_B |= other_B;
            return Connected_A && Connected_B;
        }
    };

    class Tile {
        std::shared_ptr<LineSegment> m_linesegment;
        Tile* m_tile_ref;
    public:
        TileState tileState;
        std::shared_ptr<LineSegment> GetLineSegment() {
            if (m_linesegment != nullptr) {
                return m_linesegment;
            } else if (m_tile_ref != nullptr) {
                return m_tile_ref->GetLineSegment();
            } else {
                return nullptr;
            }
        }

        LineSegment* ReferLineSegment(Tile* newref) {
            if (m_linesegment != nullptr) {
                m_linesegment = nullptr;
                m_tile_ref = newref;
            } else if (m_tile_ref != nullptr) {
                m_tile_ref->ReferLineSegment(newref);
            }
        }

        bool OwnsLine() { return m_linesegment != nullptr; }

        Tile () {
            tileState = Empty;
            m_linesegment = nullptr;
            m_tile_ref = nullptr;
        }

        bool PlaceTile(TileState newState, Tile* neighbours[6], int edge_id) {
            tileState = newState;
            //std::cout << edge_id << std::endl;
            bool tile_connected_A = (edge_id == -1);
            bool tile_connected_B = (edge_id == 1);

            bool foundNeighbour = false;
            bool won = false;
            std::shared_ptr<LineSegment> neighbourLineSegment = nullptr;
            std::shared_ptr<LineSegment> myLineSegment = nullptr;
            for (int i=0; i < 6; i++) {
                if (neighbours[i] != nullptr && neighbours[i]->tileState == tileState) {
                    myLineSegment = GetLineSegment();
                    neighbourLineSegment = neighbours[i]->GetLineSegment();
                    if (foundNeighbour && neighbourLineSegment != myLineSegment) {
                        // Another neighbour, combine segments
                        won |= neighbourLineSegment->Merge(myLineSegment);
                        ReferLineSegment(neighbours[i]);
                    }
                    else if (!foundNeighbour) {
                        // First neighbour, add to this segment
                        foundNeighbour = true;
                        won |= neighbourLineSegment->Merge(tile_connected_A, tile_connected_B);
                        m_tile_ref = neighbours[i];
                    }
                }
            }
            if (!foundNeighbour) {
                m_linesegment = std::shared_ptr<LineSegment>(new LineSegment(tile_connected_A, tile_connected_B));
            }
            return won;
        }
    };

    class Game {

	    shark::blas::matrix<Tile> m_gameboard;
        unsigned m_current_player = 0;

        const std::string m_red_color = "\033[1;31m";
        const std::string m_blue_color = "\033[1;34m";
        const std::string m_reset_color = "\033[0;0m";
        const std::string m_hexes[3] = {
            m_blue_color + "\u2b22" + m_reset_color, // Blue
            m_red_color + "\u2b22" + m_reset_color,  // Red
            "\u2b21"                                 // Empty
        };

        bool m_one_player_debug_mode = false;

        void m_next_player() {
            if (!m_one_player_debug_mode) {
                m_current_player = (m_current_player + 1) % 2;
            }
        }

        void m_initializeboard() {
            for (int i=0; i < BOARD_SIZE; i++) {
                for (int j=0; j < BOARD_SIZE; j++) {
                    m_gameboard(i,j) = Tile();
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

        Tile* m_get_neighbour(int x, int y) {
            if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE || m_gameboard(x,y).tileState == Empty) {
                return nullptr;
            }
            return &m_gameboard(x,y);
        }

        bool m_place_tile(std::pair<unsigned, unsigned> pos, bool* won) {
            Tile* neighbours[6] = {
                // Upper neigbours
                m_get_neighbour(pos.first,   pos.second-1),
                m_get_neighbour(pos.first+1, pos.second-1),
                // Horizontal neighbours
                m_get_neighbour(pos.first-1, pos.second),
                m_get_neighbour(pos.first+1, pos.second),
                // Lower neighbours
                m_get_neighbour(pos.first-1, pos.second+1),
                m_get_neighbour(pos.first,   pos.second+1)
            };
            int edge_id = 0;
            switch (m_current_player) {
                case 0:
                    if      (pos.second == 0) { edge_id = -1; }
                    else if (pos.second == BOARD_SIZE-1) { edge_id = 1; }
                    break;
                case 1:
                    if      (pos.first == 0) { edge_id = -1; }
                    else if (pos.first == BOARD_SIZE-1) { edge_id = 1; }
                    break;
            }
            return m_gameboard(pos.first, pos.second).PlaceTile((TileState)m_current_player, neighbours, edge_id);
        }

        std::pair<unsigned,unsigned> m_alphnum2num(std::string position) {
            return std::pair<unsigned,unsigned>(atoi(position.substr(1, position.length()-1).c_str()) - 1, toupper(position[0]) - 65);
        }

    public:
        Game(bool test_mode) : m_gameboard(BOARD_SIZE, BOARD_SIZE) {
            m_initializeboard();
            m_one_player_debug_mode = test_mode;
        }
        static const unsigned BOARD_SIZE = 11;

        unsigned CurrentPlayer() {return m_current_player;}

        bool take_turn(std::string pos, bool* won) {
            std::pair<unsigned,unsigned> pos_pair = m_alphnum2num(pos);
            if (m_gameboard(pos_pair.first, pos_pair.second).tileState != Empty) {
                return false;
            }
            *won = m_place_tile(pos_pair, won);
            m_next_player();
            return true;
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
                    std::cout << " " << m_hexes[m_gameboard(i, j).tileState];
                }
                std::cout << " " << m_blue_color << i+1 << m_reset_color;
                std::cout << std::endl;
            }
            m_printletters(BOARD_SIZE + 2);
        }

        void print_segments() {
            for (int i=0; i < BOARD_SIZE; i++) {
                for (int j=0; j < BOARD_SIZE; j++) {
                    if (m_gameboard(i, j).OwnsLine()) {
                        if (m_gameboard(i,j).tileState == Red) {
                            std::cout << "Red " << std::endl;
                        }
                        else {
                            std::cout << "Blue " << std::endl;
                        }
                    }
                }
            }
        }
    };
}

#endif