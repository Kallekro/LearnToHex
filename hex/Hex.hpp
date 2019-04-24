#ifndef HEX_HPP
#define HEX_HPP
#include <shark/LinAlg/Base.h>
#include <string>
#include <memory>

namespace Hex {

    static const unsigned BOARD_SIZE = 11;

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

    class Strategy{
    public:
        virtual shark::RealVector getMoveAction(shark::blas::matrix<char>const& field) = 0;
        virtual std::size_t numParameters() const = 0;
        virtual void setParameters(shark::RealVector const& parameters) = 0;
    };


    class Game {

	    shark::blas::matrix<Tile> m_gameboard;
        unsigned m_activePlayer = 0;

        const std::string m_red_color = "\033[1;31m";
        const std::string m_blue_color = "\033[1;34m";
        const std::string m_reset_color = "\033[0;0m";
        const std::string m_hexes[3] = {
            m_blue_color + "\u2b22" + m_reset_color, // Blue
            m_red_color + "\u2b22" + m_reset_color,  // Red
            "\u2b21"                                 // Empty
        };

        bool m_one_player_debug_mode = false;

        void m_initializeboard() {
            for (int i=0; i < BOARD_SIZE; i++) {
                for (int j=0; j < BOARD_SIZE; j++) {
                    m_gameboard(i,j) = Tile();
                }
            }
        }

        void m_next_player() {
            if (!m_one_player_debug_mode) {
                m_activePlayer = (m_activePlayer + 1) % 2;
            }
        }

        shark::IntVector feasibleMoveActions(shark::blas::matrix<Tile> const &field) {
            shark::IntVector feasible(BOARD_SIZE * BOARD_SIZE, 1);
            return feasible;
        }

        shark::RealVector toFeasibleProbabilities(shark::RealVector probs, shark::IntVector const& feasible) const {
            return probs;
        }

        unsigned sampleAction(shark::RealVector const& actionProb) const {
            return actionProb.size() - 1;
        }

        bool takeMoveAction(unsigned moveAction) {
            std::pair<unsigned, unsigned> pos (0, 0);
            return m_place_tile(pos);
        }

        Tile* m_get_neighbour(int x, int y) {
            if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE || m_gameboard(x,y).tileState == Empty) {
                return nullptr;
            }
            return &m_gameboard(x,y);
        }

        bool m_place_tile(std::pair<unsigned, unsigned> pos) {
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
            switch (m_activePlayer) {
                case 0:
                    if      (pos.second == 0) { edge_id = -1; }
                    else if (pos.second == BOARD_SIZE-1) { edge_id = 1; }
                    break;
                case 1:
                    if      (pos.first == 0) { edge_id = -1; }
                    else if (pos.first == BOARD_SIZE-1) { edge_id = 1; }
                    break;
            }
            return m_gameboard(pos.first, pos.second).PlaceTile((TileState)m_activePlayer, neighbours, edge_id);
        }

        std::pair<unsigned,unsigned> m_alphnum2num(std::string position) {
            return std::pair<unsigned,unsigned>(atoi(position.substr(1, position.length()-1).c_str()) - 1, toupper(position[0]) - 65);
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

    public:
        typedef Hex::Strategy Strategy;

        Game(bool test_mode) : m_gameboard(BOARD_SIZE, BOARD_SIZE) {
            m_initializeboard();
            m_one_player_debug_mode = test_mode;
        }

        unsigned ActivePlayer() {return m_activePlayer;}

        bool takeTurn(std::vector<Strategy*> const& strategies) {
            m_next_player();
            // get player information
            auto strategy = strategies[m_activePlayer];
            // find all feasible moves
            auto feasibleMoves = feasibleMoveActions(m_gameboard);
            // get action preferences from player and transform into probabilities
            shark::RealVector moveProbs = toFeasibleProbabilities(strategy->getMoveAction(m_gameboard), feasibleMoves);
            // sample an action and take turn
            double moveAction = sampleAction(moveProbs);
            bool won = takeMoveAction(moveAction);

            return won;
        }

        // Manual turn for playing the game without RL strategy
        bool takeManualTurn(std::string pos, bool* won) {
            std::pair<unsigned,unsigned> pos_pair = m_alphnum2num(pos);
            if (m_gameboard(pos_pair.first, pos_pair.second).tileState != Empty) {
                return false;
            }
            *won = m_place_tile(pos_pair);
            m_next_player();
            return true;
        }

        void printhex() {
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
/*
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
*/
    };

    //random strategy for testing
    class RandomStrategy: public Strategy{
    public:
        shark::RealVector getMoveAction(shark::blas::matrix<char>const&) override{
            return shark::RealVector(BOARD_SIZE * BOARD_SIZE, 1.0);
        }

        std::size_t numParameters() const override{
            return 0;
        }

        void setParameters(shark::RealVector const&) override{}
    };
}

#endif