import copy
from new_game_dialog import Ui_Dialog
from poker_table import Ui_MainWindow
from PySide2.QtWidgets import QDialog, QApplication, QMainWindow, QLabel, QLineEdit, QSlider
from PySide2.QtGui import QPixmap, QDoubleValidator
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile
from PySide2.QtCore import Qt, QTimer
import sys
import time
from PokerGame.NLHoldem import Game, value_dict, suit_dict
class GameSetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.okButton.clicked.connect(self.accept)
        self.ui.cancelButton.clicked.connect(self.reject)

    def get_inputs(self):
        return {
            'num_players': self.ui.spinBoxNumPlayers.value(),
            'starting_stack': self.ui.spinBoxStartingStack.value(),
            'username': self.ui.lineEditUsername.text()
        }

class PokerWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, username, game_instance, parent=None, n_players = 9):
        super(PokerWindow, self).__init__(parent)
        self.game = game_instance
        self.setupUi(self)  # this sets up the GUI as defined in Qt Designer
        self.labelPlayerName_0.setText(username)
        # Connect signals to your slots here
        self.foldButton.clicked.connect(self.fold)
        self.checkCallButton.clicked.connect(self.call)
        self.betRaiseButton.clicked.connect(self.raiseBet)
        self.raiseAmountLineEdit = self.findChild(QLineEdit, 'raiseAmountLineEdit')
        # Sync the QLineEdit with the QSlider

        self.raiseAmountLineEdit = self.findChild(QLineEdit, 'raiseAmountLineEdit')
        self.betSizeSlider = self.findChild(QSlider, 'betSizeSlider')
        self.raiseAmountLineEdit.textEdited.connect(self.onRaiseAmountEdited)
        self.betSizeSlider.valueChanged.connect(self.onSliderValueChanged)

        # Set up a double validator with a range and two decimal places
        double_validator = QDoubleValidator(0.1, 10000.0, 2, self.raiseAmountLineEdit)
        double_validator.setNotation(QDoubleValidator.StandardNotation)
        self.raiseAmountLineEdit.setValidator(double_validator)



        self.player_elements = {
            0: {"widget": self.player0, 'cards': [self.holecardP00, self.holecardP01], "button": self.button0, "bet": self.bet_0},
            1: {"widget": self.player1, 'cards': [self.holecardP10, self.holecardP11], "button": self.button1, "bet": self.bet_1},
            2: {"widget": self.player2, 'cards': [self.holecardP20, self.holecardP21], "button": self.button2, "bet": self.bet_2},
            3: {"widget": self.player3, 'cards': [self.holecardP30, self.holecardP31], "button": self.button3, "bet": self.bet_3},
            4: {"widget": self.player4, 'cards': [self.holecardP40, self.holecardP41], "button": self.button4, "bet": self.bet_4},
            5: {"widget": self.player5, 'cards': [self.holecardP50, self.holecardP51], "button": self.button5, "bet": self.bet_5},
            6: {"widget": self.player6, 'cards': [self.holecardP60, self.holecardP61], "button": self.button6, "bet": self.bet_6},
            7: {"widget": self.player7, 'cards': [self.holecardP70, self.holecardP71], "button": self.button7, "bet": self.bet_7},
            8: {"widget": self.player8, 'cards': [self.holecardP80, self.holecardP81], "button": self.button8, "bet": self.bet_8},
        }

        self._adjustTableForLessPlayers(n_players)

        # generate dict for all geometries
        self.geometries = {i: {} for i in range(n_players)}
        for i in range(n_players):
            elements = self.player_elements[i].items()
            for key, value in elements:
                if type(value) != list:
                    self.geometries[i][key] = copy.deepcopy(value.geometry())
                else:
                    self.geometries[i][key] = []
                    for j in range(len(value)):
                        self.geometries[i][key].append(copy.deepcopy(value[j].geometry()))


    def onRaiseAmountEdited(self, text):
        # Convert text to a float value and update the slider's position
        value = float(text) if text else 0
        self.betSizeSlider.setValue(int(value))  # Assumes the slider uses integer values

    def onSliderValueChanged(self, value):
        # Update the line edit whenever the slider value changes
        self.raiseAmountLineEdit.setText(f"{value:.2f}")  # Assuming you want to show float values with two decimal places


    def hidePlayerElements(self, player):
        # Hide all player elements
        elements = self.player_elements[player].items()
        for key, value in elements:
            if type(value) != list:
                value.setVisible(False)
            else:
                for element in value:
                    element.setVisible(False)

    def _switch_positions(self, from_player, to_player):
        elements_from = self.player_elements[from_player]
        elements_to = self.player_elements[to_player].items()
        for key, value in elements_to:
            if type(value) != list:
                position = value.geometry()
                elements_from[key].setGeometry(position)
            else:
                for i in range(len(value)):
                    position = value[i].geometry()
                    elements_from[key][i].setGeometry(position)


    def _adjustTableForLessPlayers(self, n_players):
        # Reposition and hide logic here
        players_hide = list(range(n_players,9))
        for player in players_hide:
            self.hidePlayerElements(player)
        if n_players == 2:
            self._switch_positions(1,5)
        elif n_players == 3:
            self._switch_positions(1, 4)
            self._switch_positions(2, 6)
        elif n_players == 4:
            self._switch_positions(1, 2)
            self._switch_positions(2, 5)
            self._switch_positions(3, 7)
        elif n_players == 5:
            self._switch_positions(1, 2)
            self._switch_positions(2, 4)
            self._switch_positions(3, 6)
            self._switch_positions(4, 7)
        elif n_players == 6:
            self._switch_positions(1, 2)
            self._switch_positions(2, 3)
            self._switch_positions(3, 5)
            self._switch_positions(4, 7)
            self._switch_positions(5, 8)
        elif n_players == 7:
            self._switch_positions(1, 2)
            self._switch_positions(2, 3)
            self._switch_positions(3, 4)
            self._switch_positions(4, 6)
            self._switch_positions(5, 7)
            self._switch_positions(6, 8)
        elif n_players == 8:
            self._switch_positions(1, 2)
            self._switch_positions(2, 3)
            self._switch_positions(3, 4)
            self._switch_positions(4, 5)
            self._switch_positions(5, 6)
            self._switch_positions(6, 7)
            self._switch_positions(7, 8)

    def set_geometry(self, player, seat):
        geometries = self.geometries[seat]
        player_elements = self.player_elements[player]
        for key, value in player_elements.items():
            if type(value) != list:
                position = geometries[key]
                value.setGeometry(position)
            else:
                for i in range(len(value)):
                    position = geometries[key][i]
                    value[i].setGeometry(position)


    def setCardImage(self, label, card):
        # Construct the filename from the card's value and suit properties
        card_filename = f"{value_dict[card.value] if card.value >= 10 else card.value}{suit_dict[card.suit]}.png"
        card_path = f"cards/{card_filename}"  # Replace with the correct path
        pixmap = QPixmap(card_path)
        scaled_pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        label.setVisible(True)

    def showCards(self, player):
        self.setCardImage(self.player_elements[player]["cards"][0], self.game.players[player].holecards[0])
        self.player_elements[player]["cards"][0].setVisible(True)
        self.setCardImage(self.player_elements[player]["cards"][1], self.game.players[player].holecards[1])
        self.player_elements[player]["cards"][1].setVisible(True)
    def muckCards(self, player):
        self.player_elements[player]["cards"][0].setVisible(False)
        self.player_elements[player]["cards"][1].setVisible(False)

    # Define your slot functions here
    def fold(self):
        current_street = self.game.street
        # handle fold action
        index_next = self.game.players.index(self.game.next[-1])
        self.muckCards(index_next)
        game.implement_action(self.game.next[-1], 0)
        if self.game.street > current_street and not self.game.finished:
            self.deal_board(current_street)


        self._set_buttons()
        self.check_end_of_round(current_street)


    def call(self):
        current_street = self.game.street
        player = self.game.next[-1]
        index_next = self.game.players.index(player)

        call_amount = self.game.max_bet - player.bet

        game.implement_action(self.game.next[-1], 1)
        bet_label = self.player_elements[index_next]["bet"]
        bet_label.setText(str(player.bet))
        if player.bet != 0:
            bet_label.setVisible(True)
        else:
            bet_label.setVisible(False)

        chipcount_label = mainWin.player_elements[index_next]["widget"].findChild(QLabel, f"chipcountPlayer_{index_next}")
        current_bet_label = chipcount_label.text()
        new_stack_label = float(current_bet_label) - call_amount
        if int(new_stack_label) == float(new_stack_label):
            new_stack_label = int(new_stack_label)
        chipcount_label.setText(str(new_stack_label))

        if self.game.street > current_street and not self.game.finished:
            self.deal_board(current_street)

        self._set_buttons()
        self.check_end_of_round(current_street)

    def raiseBet(self):
        raise_amount = float(self.raiseAmountLineEdit.text())
        player = self.game.next[-1]
        index_next = self.game.players.index(player)

        current_bet = player.bet
        game.implement_action(player, 2, raise_amount)

        bet_label = self.player_elements[index_next]["bet"]
        bet_label.setText(str(raise_amount))
        if player.bet != 0:
            bet_label.setVisible(True)
        else:
            bet_label.setVisible(False)

        change_in_bet = raise_amount - current_bet
        chipcount_label = mainWin.player_elements[index_next]["widget"].findChild(QLabel, f"chipcountPlayer_{index_next}")
        current_bet_label = chipcount_label.text()
        new_stack_label = float(current_bet_label) - change_in_bet
        if int(new_stack_label) == float(new_stack_label):
            new_stack_label = int(new_stack_label)
        chipcount_label.setText(str(new_stack_label))

        self._set_buttons()
        self.check_end_of_round()
    def deal_board(self, street):
        self.potLabel.setText("Pot: " + str(self.game.pot))
        for value in self.player_elements.values():
            value["bet"].setVisible(False)
        # street is the current street before dealing, so if flop is dealt, street i 0
        if street == 0:
            mainWin.setCardImage(self.flop_0, self.game.board[0])
            mainWin.setCardImage(self.flop_1, self.game.board[1])
            mainWin.setCardImage(self.flop_2, self.game.board[2])
        elif street == 1:
            mainWin.setCardImage(self.turn, self.game.board[1])

        else:
            mainWin.setCardImage(self.river, self.game.board[2])


    def _set_buttons(self):
        # change labels on buttons for next player
        if len(self.game.next) != 0:
            if self.game.max_bet == 0:
                self.betRaiseButton.setText("Bet")
            else:
                self.betRaiseButton.setText("Raise")

            if self.game.next[-1].bet == self.game.max_bet:
                self.checkCallButton.setText("Check")
                self.foldButton.setEnabled(False)
            else:
                self.checkCallButton.setText("Call")
                self.foldButton.setEnabled(True)

            # Assuming betSizeSlider is your QSlider object
            minbet = self.game.get_legal_betsize()
            maxbet = self.game.next[-1].stack + self.game.next[-1].bet
            self.betSizeSlider.setMinimum(minbet)  # Set the slider's minimum value to 10
            self.betSizeSlider.setMaximum(maxbet)  # Set the slider's maximum value to 1000

    def check_end_of_round(self, street=None):
        if self.game.finished:
            if len(self.game.left_in_hand) > 1:
                print("showdown")
                # Queue the UI updates for dealing cards with delays in between
                if street == 0 or street is None:  # If street is None, deal all streets
                    QTimer.singleShot(1000, lambda: self.deal_board(0))  # Deal flop after 1 sec
                    QTimer.singleShot(2000, lambda: self.deal_board(1))  # Deal turn after 2 secs
                    QTimer.singleShot(3000, lambda: self.deal_board(2))  # Deal river after 3 secs

                elif street == 1:
                    QTimer.singleShot(1000, lambda: self.deal_board(1))  # Deal turn after 1 sec
                    QTimer.singleShot(2000, lambda: self.deal_board(2))  # Deal river after 2 secs

                elif street == 2:
                    QTimer.singleShot(1000, lambda: self.deal_board(2))  # Deal river after 1 sec

            wait_till_chip_update = 4000 - street * 1000 if street is not None else 1000
                # Queue the chip count updates to occur after the last card is dealt
            QTimer.singleShot(wait_till_chip_update, self.update_chip_counts)  # Update chip counts after all cards are dealt
            wait_till_new_hand = 6000 - street * 1000 if street is not None else 1000
            QTimer.singleShot(wait_till_new_hand, self.deal_new_hand)



    def update_chip_counts(self):
        # This function will update the chip counts after the board has been fully dealt
        for i in range(self.game.n_players):  # Assuming 'n_players' is defined
            chipcount_label = self.player_elements[i]["widget"].findChild(QLabel, f"chipcountPlayer_{i}")
            chipcount_label.setText(str(self.game.players[i].stack))
            self.player_elements[i]["bet"].setText(str(0))
            self.player_elements[i]["bet"].setVisible(False)



    def deal_new_hand(self):
        self.game.new_hand()
        index_hero = self.game.positions.index(self.game.players[0])

        for i in range(n_players):
            # find position player_0
            position = self.game.positions.index(self.game.players[i]) - index_hero
            if position < 0:
                position += n_players
            self.set_geometry(i, position)

        player_button = self.game.next[-1] if n_players == 2 else self.game.positions[-1]
        index_button = self.game.players.index(player_button)
        for i in range(n_players):
            if i != index_button:
                self.player_elements[i]["button"].setVisible(False)
        for i in range(n_players):
            self.showCards(i)
            chipcount_label = self.player_elements[i]["widget"].findChild(QLabel, f"chipcountPlayer_{i}")
            chipcount_label.setText(str(self.game.players[i].stack))
            bet = self.game.players[i].bet
            if bet > 0:
                self.player_elements[i]["bet"].setText(str(game.players[i].bet))
                self.player_elements[i]["bet"].setVisible(True)
        self.flop_0.setVisible(False)
        self.flop_1.setVisible(False)
        self.flop_2.setVisible(False)
        self.turn.setVisible(False)
        self.river.setVisible(False)
        self.potLabel.setText("Pot: " + str(self.game.pot))
        mainWin._set_buttons()



def init_game(game, mainWin):
    game.new_hand(first_hand=True, random_seat=True)

    # seat players according to order in game-instance. Player_0 always sits in the first seat because hero
    index_hero = game.positions.index(game.players[0])
    positions = [game.positions.index(game.players[i]) for i in range(n_players)]
    indices_next = [game.players.index(game.next[i]) for i in range(n_players)]
    print(positions)
    print(indices_next)
    for i in range(n_players):
        # find position player_0
        position = game.positions.index(game.players[i]) - index_hero
        if position < 0:
            position += n_players
        mainWin.set_geometry(i, position)


    player_button = game.next[-1] if n_players == 2 else game.positions[-1]
    index_button = game.players.index(player_button)
    for i in range(n_players):
        if i != index_button:
            mainWin.player_elements[i]["button"].setVisible(False)
    for i in range(n_players):
        mainWin.showCards(i)
        chipcount_label = mainWin.player_elements[i]["widget"].findChild(QLabel, f"chipcountPlayer_{i}")
        chipcount_label.setText(str(game.players[i].stack))
        bet = game.players[i].bet
        if bet > 0:
            mainWin.player_elements[i]["bet"].setText(str(game.players[i].bet))
    mainWin._set_buttons()

if __name__ == "__main__":
    app = QApplication([])

    dialog = GameSetupDialog()
    if dialog.exec_():
        game_settings = dialog.get_inputs()
        n_players = game_settings["num_players"]
        starting_stacks = [game_settings["starting_stack"] for i in range(n_players)]

        print(game_settings)  # For now, just to confirm it's working
        game = Game(n_players, starting_stacks, )
        mainWin = PokerWindow(game_settings["username"], n_players=n_players, game_instance=game)


        init_game(game, mainWin)
        mainWin.show()
    app.exec_()


