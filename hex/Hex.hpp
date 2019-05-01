#ifndef HEX_HPP
#define HEX_HPP
#include <shark/LinAlg/Base.h>
#include <string>
#include <memory>

namespace Hex {
    static const unsigned BOARD_SIZE = 5;

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

        Tile () {
            tileState = Empty;
            m_linesegment = nullptr;
            m_tile_ref = nullptr;
        }

        void reset() {
            tileState = Empty;
            m_linesegment = nullptr;
            m_tile_ref = nullptr;
        }

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
        virtual shark::RealVector getMoveAction(shark::blas::matrix<Tile>const& field) = 0;
        virtual std::size_t numParameters() const = 0;
        virtual void setParameters(shark::RealVector const& parameters) = 0;
    };

    //random strategy for testing
    class RandomStrategy: public Strategy{
    public:
        shark::RealVector getMoveAction(shark::blas::matrix<Tile>const&) override{
            return shark::RealVector(BOARD_SIZE * BOARD_SIZE, 1.0);
        }

        std::size_t numParameters() const override{
            return 1;
        }

        void setParameters(shark::RealVector const&) override{}
    };

    class Game {

	    shark::blas::matrix<Tile> m_gameboard;
        unsigned m_activePlayer = 0;
        unsigned m_playerWon = -1;


        const std::string m_red_color = "\033[1;31m";
        const std::string m_blue_color = "\033[1;34m";
        const std::string m_reset_color = "\033[0;0m";
        const std::string m_hexes[3] = {
            m_blue_color + "\u2b22" + m_reset_color, // Blue
            m_red_color + "\u2b22" + m_reset_color,  // Red
            "\u2b21"                                 // Empty
        };

        bool m_one_player_debug_mode = false;

        // Variables for termination
        std::vector<int> m_player_ranks;
        unsigned int m_next_rank;

        // log for importance sampling
        struct Log {
            unsigned current_player;
            unsigned x;
            unsigned y;
            double logprob;
            unsigned moveAction;
            shark::blas::matrix<Tile> move_state;
        };
        Log m_laststep;

        shark::IntVector m_feasible_move_actions(shark::blas::matrix<Tile> const& field) const {
            shark::IntVector feasible_moves((BOARD_SIZE*BOARD_SIZE), 1);
            for (int i=0; i<BOARD_SIZE; i++) {
                for (int j=0; j<BOARD_SIZE; j++) {
                    if (m_gameboard(i, j).tileState != Empty) {
                        feasible_moves(BOARD_SIZE * i + j) = 0;
                    }
                }
            }
            return feasible_moves;
        }
        shark::RealVector m_feasible_probabilies(shark::RealVector probs, shark::IntVector const& feasible_moves) const {
            for (std::size_t i=0; i != probs.size(); i++) {
                if (!feasible_moves(i)) {
                    probs(i) = -std::numeric_limits<double>::max();
                }
            }
            probs = exp(probs - max(probs));
            probs /= sum(probs);
            return probs;
        }

        unsigned int m_sample_move_action(shark::RealVector const& action_prob)const{
            double u = shark::random::uni(shark::random::globalRng(), 0.0, 1.0);
		    double cumulant = 0.0;
            for (std::size_t i=0; i != action_prob.size(); i++) {
                cumulant += action_prob(i);
                if (cumulant > u) {
                    return i;
                }
            }
            return action_prob.size() - 1;
        }

        bool m_take_move_action(unsigned move_action) {
            unsigned x = move_action % BOARD_SIZE;
            unsigned y = move_action / BOARD_SIZE;
            std::pair<unsigned,unsigned> pos (y, x);
            return m_place_tile(pos);
        }

        void m_next_player() {
            if (!m_one_player_debug_mode) {
                m_activePlayer = (m_activePlayer + 1) % 2;
            }
        }
        void m_initializeboard() {
            for (int i=0; i < BOARD_SIZE; i++) {
                for (int j=0; j < BOARD_SIZE; j++) {
                    m_gameboard(i,j) = Tile();
                }
            }
        }

        Tile* m_get_neighbour(int x, int y) {
            if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE || m_gameboard(x,y).tileState == Empty) {
                return nullptr;
            }
            return &m_gameboard(x,y);
        }

        bool m_place_tile(std::pair<unsigned, unsigned> pos) {
            if (m_gameboard(pos.first, pos.second).tileState != Empty) {
                throw std::invalid_argument("double place!");
            }
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

        std::string m_printletters(int start_spaces) {
            std::string resStr("");
            for (int i = 0; i < start_spaces; i++) {
                resStr += " ";
            }
            for (char i = 65; i < 65+BOARD_SIZE; i++) {
                resStr += m_red_color + i + m_reset_color + " ";
            }
            resStr += '\n';
            return resStr;
        }

    public:
        typedef Hex::Strategy Strategy;

        unsigned turns_taken = 0;

        Game(bool test_mode) : m_gameboard(BOARD_SIZE, BOARD_SIZE) {
            m_initializeboard();
            m_one_player_debug_mode = test_mode;
        }


        void reset() {
            // reset board
            for (std::size_t i = 0; i < BOARD_SIZE; i++) {
                for (std::size_t j = 0; j < BOARD_SIZE; j++) {
                    m_gameboard(i,j).reset();
                }
            }
            // TODO: Choose random starter
            m_activePlayer = 1;
            m_playerWon = -1;
            turns_taken = 0;
        }

        void FlipBoard() {
            int ri = BOARD_SIZE-1;
            int rj = BOARD_SIZE-1;
            for (int i=0; i < BOARD_SIZE; i++) {
                rj = BOARD_SIZE-1;
                for (int j=0; j < BOARD_SIZE; j++) {
                    Tile tmptile = m_gameboard(i,j);

                    m_gameboard(i,j) = m_gameboard(ri, rj);
                    m_gameboard(ri,rj) = tmptile;
                    rj--;
                }
                ri--;
            }
        }

        unsigned ActivePlayer() {return m_activePlayer;}

        bool takeTurn(std::vector<Strategy*> const& strategies) {
            m_next_player();
            turns_taken++;
            // get player information
            auto strategy = strategies[m_activePlayer];
            // find all feasible moves
            auto feasibleMoves = m_feasible_move_actions(m_gameboard);
            // get action preferences from player and transform into probabilities
            shark::RealVector moveProbs = m_feasible_probabilies(strategy->getMoveAction(m_gameboard), feasibleMoves);
            // sample an action and take turn
            double moveAction = m_sample_move_action(moveProbs);
            bool won;
            try {
                won = m_take_move_action(moveAction);
            } catch (std::invalid_argument& e) {
                std::cerr << "exception: " << e.what() << std::endl;
                std::cout << feasibleMoves << std::endl;
                std::cout << moveAction << std::endl;
                std::cout << moveProbs << std::endl;
                throw(e);
            }
            if (won) {
                m_playerWon = m_activePlayer;
            }
            return !won;
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

        int getRank(std::size_t player)const {
            return player == m_playerWon ? 1 : 0;
        }

        std::string asciiState() {
            std::string resStr("");
            resStr += m_printletters(1);
            for (int i=0; i < BOARD_SIZE; i++) {
                for (int ii=0; ii < i; ii++) {
                    if (ii == 8) { continue; }
                    resStr += " ";
                }
                resStr += m_blue_color + std::to_string(i+1) + m_reset_color;
                for (int j=0; j < BOARD_SIZE; j++) {
                    resStr += " " + m_hexes[m_gameboard(i, j).tileState];
                }
                resStr +=  " " + m_blue_color + std::to_string(i+1) + m_reset_color;
                resStr += '\n';
            }
            resStr += m_printletters(BOARD_SIZE + 2);
            return resStr;
        }
    };
}

#endif