import torch
import torch.nn as nn



class PokerHandEmbedding(nn.Module):
    def __init__(self, embedding_dim, feature_dim, deep_layer_sizes, intermediary_dim):
        """
        Initialize the PokerHandEmbedding model with dynamically created deep layers for each street.

        Args:
            embedding_dim (int): Dimension of card embeddings.
            feature_dim (int): Dimension of feature representations.
            deep_layer_sizes (tuple): Tuple of sizes for deep layers (preflop, flop, turn, river).
            intermediary_dim (int): Dimension of intermediate layers in task-specific heads.
        """
        super(PokerHandEmbedding, self).__init__()
        self.card_embedding = nn.Embedding(52, embedding_dim)
        self.feature_dim = feature_dim

        # Dynamically create deep layers based on the sizes
        self.preflop_deep_layer = nn.Linear(embedding_dim * 2, deep_layer_sizes[0])
        self.flop_deep_layer = nn.Linear(feature_dim + embedding_dim * 3 + 6, deep_layer_sizes[1])
        self.turn_deep_layer = nn.Linear(feature_dim + embedding_dim + 85, deep_layer_sizes[2])
        self.river_deep_layer = nn.Linear(feature_dim + embedding_dim + 221, deep_layer_sizes[3])

        # Fully connected layers for feature extraction
        self.preflop_fc = nn.Linear(deep_layer_sizes[0], feature_dim)
        self.flop_fc = nn.Linear(deep_layer_sizes[1], feature_dim)
        self.turn_fc = nn.Linear(deep_layer_sizes[2], feature_dim)
        self.river_fc = nn.Linear(deep_layer_sizes[3], feature_dim)

        # Activation functions
        self.activation = nn.GELU()


        # Task-specific heads with intermediary layers
        self.head_outcome_probs_preflop = nn.Sequential(
            nn.Linear(feature_dim, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 3)
        )
        self.head_preflop_type = nn.Sequential(
            nn.Linear(feature_dim, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 3)
        )

        self.head_hand_type_flop = nn.Sequential(
            nn.Linear(feature_dim, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 9)
        )
        self.head_hand_type_turn = nn.Sequential(
            nn.Linear(feature_dim, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 9)
        )
        self.head_hand_type_river = nn.Sequential(
            nn.Linear(feature_dim, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 9)
        )

        self.head_straight_draw_flop = nn.Sequential(
            nn.Linear(feature_dim, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 3)
        )
        self.head_straight_draw_turn = nn.Sequential(
            nn.Linear(feature_dim, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 3)
        )
        self.head_flush_draw_flop = nn.Sequential(
            nn.Linear(feature_dim, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 3)
        )
        self.head_flush_draw_turn = nn.Sequential(
            nn.Linear(feature_dim, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 3)
        )

        self.head_outcome_probs_river = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )
        # modules to evaluate board cards only without hole
        self.head_strength_flop = nn.Sequential(
            nn.Linear(feature_dim, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 3) # 0 high card 1 pair, 2 trips,
        )

        self.head_strength_turn = nn.Sequential(
            nn.Linear(feature_dim, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 5) # 0 high card 1 pair, 2 two pair, 3 trips, 4 quads
        )

        self.head_strength_river = nn.Sequential(
            nn.Linear(feature_dim, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 9)
        )

        self.head_outcome_probs_flop = nn.Sequential(
            nn.Linear(feature_dim, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 3)
        )

        self.head_outcome_probs_turn = nn.Sequential(
            nn.Linear(feature_dim, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 3)
        )

        self.head_flop_kicker_0 = nn.Sequential(
            nn.Linear(feature_dim + 9, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 13)
        )

        self.head_flop_kicker_1 = nn.Sequential(
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 13)
        )

        self.head_flop_kicker_2 = nn.Sequential(
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 13)
        )

        self.head_flop_kicker_3 = nn.Sequential( # highest possible kicker Jack
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 11)
        )

        self.head_flop_kicker_4 = nn.Sequential( # High Card and FLush have 5 kickers
            nn.Linear(feature_dim + 11, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 8) # highest possible last kicker is 9
        )


        self.head_turn_kicker_0 = nn.Sequential(
            nn.Linear(feature_dim + 9, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 13)
        )

        self.head_turn_kicker_1 = nn.Sequential(
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 13)
        )

        self.head_turn_kicker_2 = nn.Sequential(
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 13)
        )

        self.head_turn_kicker_3 = nn.Sequential( # highest possible kicker Jack
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 11)
        )

        self.head_turn_kicker_4 = nn.Sequential( # High Card and FLush have 5 kickers
            nn.Linear(feature_dim + 11, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 8) # highest possible last kicker is 9
        )


        self.head_river_kicker_0 = nn.Sequential(
            nn.Linear(feature_dim + 9, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 13)
        )

        self.head_river_kicker_1 = nn.Sequential(
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 13)
        )

        self.head_river_kicker_2 = nn.Sequential(
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 13)
        )

        self.head_river_kicker_3 = nn.Sequential( # highest possible kicker Jack
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 11)
        )

        self.head_river_kicker_4 = nn.Sequential( # High Card and FLush have 5 kickers
            nn.Linear(feature_dim + 11, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 8) # highest possible last kicker is 9
        )


        # kickers for board only
        self.head_board_turn_kicker_0 = nn.Sequential(
            nn.Linear(feature_dim + 5, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 13)
        )

        self.head_board_turn_kicker_1 = nn.Sequential(
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 13)
        )

        self.head_board_turn_kicker_2 = nn.Sequential(
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 12)
        )

        self.head_board_turn_kicker_3 = nn.Sequential( # highest possible kicker Jack
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 11)
        )



        self.head_board_river_kicker_0 = nn.Sequential(
            nn.Linear(feature_dim + 9, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 13)
        )

        self.head_board_river_kicker_1 = nn.Sequential(
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 13)
        )

        self.head_board_river_kicker_2 = nn.Sequential(
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 13)
        )

        self.head_board_river_kicker_3 = nn.Sequential( # highest possible kicker Jack
            nn.Linear(feature_dim + 13, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 11)
        )

        self.head_board_river_kicker_4 = nn.Sequential( # High Card and FLush have 5 kickers
            nn.Linear(feature_dim + 11, intermediary_dim),
            nn.GELU(),
            nn.Linear(intermediary_dim, 8) # highest possible last kicker is 9
        )

    def freeze_except(self, param_names_to_keep_trainable: list):
        """Freezes all parameters except those explicitly listed."""
        for name, param in self.named_parameters():
            param.requires_grad = any(name.startswith(p) for p in param_names_to_keep_trainable)

    def forward(
            self,
            preflop,
            flop=None,
            turn=None,
            river=None,
            flop_strength=None,
            flop_kicker_0=None,
            flop_kicker_1=None,
            flop_kicker_2=None,
            flop_kicker_3=None,
            turn_strength=None,
            turn_kicker_0 = None,
            turn_kicker_1 = None,
            turn_kicker_2 = None,
            turn_kicker_3 = None,
            river_strength = None,
            river_kicker_0 = None,
            river_kicker_1 = None,
            river_kicker_2 = None,
            river_kicker_3 = None,


            turn_strength_board=None,
            turn_kicker_board_0=None,
            turn_kicker_board_1=None,
            turn_kicker_board_2=None,
            river_strength_board=None,
            river_kicker_board_0=None,
            river_kicker_board_1=None,
            river_kicker_board_2=None,
            river_kicker_board_3=None,
    ):
        """
        Forward pass through the PokerHandEmbedding model.

        Args:
            preflop (torch.Tensor): Preflop cards (batch_size, 2).
            flop (torch.Tensor, optional): Flop cards (batch_size, 3).
            turn (torch.Tensor, optional): Turn card (batch_size, 1).
            river (torch.Tensor, optional): River card (batch_size, 1).

        Returns:
            tuple:
                torch.Tensor: Final feature vector.
                dict: A dictionary containing predicted probabilities for various poker hand outcomes and classifications.
        """
        # Initialize predictions dictionary
        predictions = {
            "log_probs_outcome_preflop": None,
            "log_probs_preflop_type": None,
            "log_probs_type_flop": None,
            "log_probs_straight_draw_flop": None,
            "log_probs_flush_draw_flop": None,
            "log_probs_type_turn": None,
            "log_probs_straight_draw_turn": None,
            "log_probs_flush_draw_turn": None,
            "log_probs_outcome_river": None,
            "log_probs_type_river": None,
            "log_probs_board_strength_flop": None,
            "log_probs_board_strength_turn": None,
            "log_probs_board_strength_river":None,
            "log_probs_outcome_flop": None,
            "log_probs_outcome_turn": None,
            "log_probs_flop_kicker_0": None,
            "log_probs_flop_kicker_1": None,
            "log_probs_flop_kicker_2": None,
            "log_probs_flop_kicker_3": None,
            "log_probs_flop_kicker_4": None,
            "log_probs_turn_kicker_0": None,
            "log_probs_turn_kicker_1": None,
            "log_probs_turn_kicker_2": None,
            "log_probs_turn_kicker_3": None,
            "log_probs_turn_kicker_4": None,
            "log_probs_river_kicker_0": None,
            "log_probs_river_kicker_1": None,
            "log_probs_river_kicker_2": None,
            "log_probs_river_kicker_3": None,
            "log_probs_river_kicker_4": None,

            "log_probs_turn_board_kicker_0": None,
            "log_probs_turn_board_kicker_1": None,
            "log_probs_turn_board_kicker_2": None,
            "log_probs_turn_board_kicker_3": None,

            "log_probs_river_board_kicker_0": None,
            "log_probs_river_board_kicker_1": None,
            "log_probs_river_board_kicker_2": None,
            "log_probs_river_board_kicker_3": None,
            "log_probs_river_board_kicker_4": None,
        }

        # Embed and process preflop cards
        preflop_emb = self.card_embedding(preflop).view(preflop.size(0), -1)
        preflop_deep = self.activation(self.preflop_deep_layer(preflop_emb))
        preflop_features = self.activation(self.preflop_fc(preflop_deep))

        # Initialize feature vector
        features = preflop_features
        predictions["log_probs_outcome_preflop"] = torch.log_softmax(self.head_outcome_probs_preflop(features), dim=-1)
        predictions["log_probs_preflop_type"] = torch.log_softmax(self.head_preflop_type(features), dim=-1)

        # Process flop stage
        if flop is not None:
            flop_emb = self.card_embedding(flop).view(flop.size(0), -1)
            # combine all flop outputs
            preflop_outputs = torch.cat(
                [
                    predictions["log_probs_outcome_preflop"].detach(),
                    predictions["log_probs_preflop_type"].detach(),
                 ], dim=-1
            )

            combined_flop = torch.cat([features, flop_emb, preflop_outputs], dim=-1)
            flop_deep = self.activation(self.flop_deep_layer(combined_flop))
            features = self.activation(self.flop_fc(flop_deep))
            predictions["log_probs_type_flop"] = torch.log_softmax(self.head_hand_type_flop(features), dim=-1)
            predictions["log_probs_straight_draw_flop"] = torch.log_softmax(self.head_straight_draw_flop(features),
                                                                            dim=-1)
            predictions["log_probs_flush_draw_flop"] = torch.log_softmax(self.head_flush_draw_flop(features), dim=-1)
            predictions["log_probs_board_strength_flop"] = torch.log_softmax(self.head_strength_flop(features), dim=-1)


            # kickers
            label_flop_value = flop_strength if flop_strength is not None else torch.argmax(predictions["log_probs_type_flop"] , dim=1)
            label_flop_value = torch.nn.functional.one_hot(label_flop_value.clamp(min=0), num_classes=9).float()
            kicker_flop_0 = torch.log_softmax(
                self.head_flop_kicker_0(torch.cat([features, label_flop_value], dim=1)), dim=-1)

            label_flop_kicker_0 = flop_kicker_0 if flop_kicker_0 is not None else torch.argmax(kicker_flop_0 , dim=1)
            label_flop_kicker_0_enc = torch.nn.functional.one_hot(label_flop_kicker_0.clamp(0), num_classes=13).float()
            label_flop_kicker_0_enc[label_flop_kicker_0==-2] = 0
            kicker_flop_1 = torch.log_softmax(
                self.head_flop_kicker_1(torch.cat([features, label_flop_kicker_0_enc], dim=1)), dim=-1)

            label_flop_kicker_1 = flop_kicker_1 if flop_kicker_1 is not None else torch.argmax(kicker_flop_1 , dim=1)
            label_flop_kicker_1_enc = torch.nn.functional.one_hot(label_flop_kicker_1.clamp(0), num_classes=13).float()
            label_flop_kicker_1_enc[label_flop_kicker_1==-2] = 0
            kicker_flop_2 = torch.log_softmax(
                self.head_flop_kicker_2(torch.cat([features, label_flop_kicker_1_enc], dim=1)), dim=-1)

            label_flop_kicker_2 = flop_kicker_2 if flop_kicker_2 is not None else torch.argmax(kicker_flop_2 , dim=1)
            label_flop_kicker_2_enc = torch.nn.functional.one_hot(label_flop_kicker_2.clamp(0), num_classes=13).float()
            label_flop_kicker_2_enc[label_flop_kicker_2==-2] = 0
            kicker_flop_3 = torch.log_softmax(
                self.head_flop_kicker_3(torch.cat([features, label_flop_kicker_2_enc], dim=1)), dim=-1)

            label_flop_kicker_3 = flop_kicker_3 if flop_kicker_3 is not None else torch.argmax(kicker_flop_3 , dim=1)
            label_flop_kicker_3_enc = torch.nn.functional.one_hot(label_flop_kicker_3.clamp(0), num_classes=11).float()
            label_flop_kicker_3_enc[label_flop_kicker_3==-2] = 0
            kicker_flop_4 = torch.log_softmax(
                self.head_flop_kicker_4(torch.cat([features, label_flop_kicker_3_enc], dim=1)), dim=-1)

            predictions.update(
                {
                    "log_probs_flop_kicker_0": kicker_flop_0,
                    "log_probs_flop_kicker_1": kicker_flop_1,
                    "log_probs_flop_kicker_2": kicker_flop_2,
                    "log_probs_flop_kicker_3": kicker_flop_3,
                    "log_probs_flop_kicker_4": kicker_flop_4,
                }
            )

            predictions["log_probs_outcome_flop"] = torch.log_softmax(self.head_outcome_probs_flop(features), dim=-1)

        # Process turn stage
        if turn is not None:
            turn_emb = self.card_embedding(turn).view(turn.size(0), -1)

            # combine all flop outputs
            flop_outputs = torch.cat(
                [
                    predictions["log_probs_type_flop"].detach(),
                    predictions["log_probs_straight_draw_flop"].detach(),
                    predictions["log_probs_flush_draw_flop"].detach(),
                    predictions["log_probs_board_strength_flop"].detach(),
                    predictions["log_probs_outcome_flop"].detach(),
                    kicker_flop_0.detach(),
                    kicker_flop_1.detach(),
                    kicker_flop_2.detach(),
                    kicker_flop_3.detach(),
                    kicker_flop_4.detach(),
                    preflop_outputs.detach(),
                ], dim=-1
            )
            combined_turn = torch.cat([features, turn_emb, flop_outputs], dim=-1)


            turn_deep = self.activation(self.turn_deep_layer(combined_turn))
            features = self.activation(self.turn_fc(turn_deep))
            predictions["log_probs_type_turn"] = torch.log_softmax(self.head_hand_type_turn(features), dim=-1)
            predictions["log_probs_straight_draw_turn"] = torch.log_softmax(self.head_straight_draw_turn(features),
                                                                            dim=-1)
            predictions["log_probs_flush_draw_turn"] = torch.log_softmax(self.head_flush_draw_turn(features), dim=-1)
            predictions["log_probs_board_strength_turn"] = torch.log_softmax(self.head_strength_turn(features), dim=-1)

            label_turn_value = turn_strength if turn_strength is not None else torch.argmax(predictions["log_probs_type_turn"] , dim=1)
            label_turn_value = torch.nn.functional.one_hot(label_turn_value.clamp(min=0), num_classes=9).float()
            kicker_turn_0 = torch.log_softmax(
                self.head_turn_kicker_0(torch.cat([features, label_turn_value], dim=1)), dim=-1)

            label_turn_kicker_0 = turn_kicker_0 if turn_kicker_0 is not None else torch.argmax(kicker_turn_0 , dim=1)
            label_turn_kicker_0_enc = torch.nn.functional.one_hot(label_turn_kicker_0.clamp(0), num_classes=13).float()
            label_turn_kicker_0_enc[label_turn_kicker_0==-2] = 0
            kicker_turn_1 = torch.log_softmax(
                self.head_turn_kicker_1(torch.cat([features, label_turn_kicker_0_enc], dim=1)), dim=-1)

            label_turn_kicker_1 = turn_kicker_1 if turn_kicker_1 is not None else torch.argmax(kicker_turn_1 , dim=1)
            label_turn_kicker_1_enc = torch.nn.functional.one_hot(label_turn_kicker_1.clamp(0), num_classes=13).float()
            label_turn_kicker_1_enc[label_turn_kicker_1==-2] = 0
            kicker_turn_2 = torch.log_softmax(
                self.head_turn_kicker_2(torch.cat([features, label_turn_kicker_1_enc], dim=1)), dim=-1)

            label_turn_kicker_2 = turn_kicker_2 if turn_kicker_2 is not None else torch.argmax(kicker_turn_2 , dim=1)
            label_turn_kicker_2_enc = torch.nn.functional.one_hot(label_turn_kicker_2.clamp(0), num_classes=13).float()
            label_turn_kicker_2_enc[label_turn_kicker_2==-2] = 0
            kicker_turn_3 = torch.log_softmax(
                self.head_turn_kicker_3(torch.cat([features, label_turn_kicker_2_enc], dim=1)), dim=-1)

            label_turn_kicker_3 = turn_kicker_3 if turn_kicker_3 is not None else torch.argmax(kicker_turn_3 , dim=1)
            label_turn_kicker_3_enc = torch.nn.functional.one_hot(label_turn_kicker_3.clamp(0), num_classes=11).float()
            label_turn_kicker_3_enc[label_turn_kicker_3==-2] = 0
            kicker_turn_4 = torch.log_softmax(
                self.head_turn_kicker_4(torch.cat([features, label_turn_kicker_3_enc], dim=1)), dim=-1)

            predictions.update(
                {
                    "log_probs_turn_kicker_0": kicker_turn_0,
                    "log_probs_turn_kicker_1": kicker_turn_1,
                    "log_probs_turn_kicker_2": kicker_turn_2,
                    "log_probs_turn_kicker_3": kicker_turn_3,
                    "log_probs_turn_kicker_4": kicker_turn_4,
                })


            label_turn_board_value = turn_strength_board if turn_strength_board is not None else torch.argmax(predictions["log_probs_board_strength_turn"] , dim=1)
            label_turn_board_value = torch.nn.functional.one_hot(label_turn_board_value.clamp(min=0), num_classes=5).float()
            kicker_turn_board_0 = torch.log_softmax(
                self.head_board_turn_kicker_0(torch.cat([features, label_turn_board_value], dim=1)), dim=-1)

            label_turn_board_kicker_0 = turn_kicker_board_0 if turn_kicker_board_0 is not None else torch.argmax(kicker_turn_board_0 , dim=1)
            label_turn_board_kicker_0_enc = torch.nn.functional.one_hot(label_turn_board_kicker_0.clamp(0), num_classes=13).float()
            label_turn_board_kicker_0_enc[label_turn_board_kicker_0==-2] = 0
            kicker_turn_board_1 = torch.log_softmax(
                self.head_board_turn_kicker_1(torch.cat([features, label_turn_board_kicker_0_enc], dim=1)), dim=-1)

            label_turn_board_kicker_1 = turn_kicker_board_1 if turn_kicker_board_1 is not None else torch.argmax(kicker_turn_board_1 , dim=1)
            label_turn_board_kicker_1_enc = torch.nn.functional.one_hot(label_turn_board_kicker_1.clamp(0), num_classes=13).float()
            label_turn_board_kicker_1_enc[label_turn_board_kicker_1==-2] = 0
            kicker_turn_board_2 = torch.log_softmax(
                self.head_board_turn_kicker_2(torch.cat([features, label_turn_board_kicker_1_enc], dim=1)), dim=-1)

            label_turn_board_kicker_2 = turn_kicker_board_2 if turn_kicker_board_2 is not None else torch.argmax(kicker_turn_board_2 , dim=1)
            label_turn_board_kicker_2_enc = torch.nn.functional.one_hot(label_turn_board_kicker_2.clamp(0), num_classes=13).float()
            label_turn_board_kicker_2_enc[label_turn_board_kicker_2==-2] = 0
            kicker_turn_board_3 = torch.log_softmax(
                self.head_board_turn_kicker_3(torch.cat([features, label_turn_board_kicker_2_enc], dim=1)), dim=-1)

            predictions.update(
                {
                    "log_probs_turn_board_kicker_0": kicker_turn_board_0,
                    "log_probs_turn_board_kicker_1": kicker_turn_board_1,
                    "log_probs_turn_board_kicker_2": kicker_turn_board_2,
                    "log_probs_turn_board_kicker_3": kicker_turn_board_3,
                })
            predictions["log_probs_outcome_turn"] = torch.log_softmax(self.head_outcome_probs_turn(features), dim=-1)


        # Process river stage
        if river is not None:
            river_emb = self.card_embedding(river).view(river.size(0), -1)

            turn_outputs = torch.cat(
                [
                    predictions["log_probs_type_turn"].detach(),
                    predictions["log_probs_straight_draw_turn"].detach(),
                    predictions["log_probs_flush_draw_turn"].detach(),
                    predictions["log_probs_board_strength_turn"].detach(),
                    predictions["log_probs_outcome_turn"].detach(),
                    kicker_turn_0.detach(),
                    kicker_turn_1.detach(),
                    kicker_turn_2.detach(),
                    kicker_turn_3.detach(),
                    kicker_turn_4.detach(),
                    kicker_turn_board_0.detach(),
                    kicker_turn_board_1.detach(),
                    kicker_turn_board_2.detach(),
                    kicker_turn_board_3.detach(),
                    preflop_outputs.detach(),
                    flop_outputs.detach(),
                ], dim=-1
            )

            combined_river = torch.cat([features, river_emb, turn_outputs], dim=-1)
            river_deep = self.activation(self.river_deep_layer(combined_river))
            features = self.activation(self.river_fc(river_deep))

            predictions["log_probs_type_river"] = torch.log_softmax(self.head_hand_type_river(features), dim=-1)
            predictions["log_probs_board_strength_river"] = torch.log_softmax(self.head_strength_river(features), dim=-1)

            # kickers
            label_river_value = river_strength if river_strength is not None else torch.argmax(predictions["log_probs_type_river"] , dim=1)
            label_river_value = torch.nn.functional.one_hot(label_river_value.clamp(min=0), num_classes=9).float()
            kicker_river_0 = torch.log_softmax(
                self.head_river_kicker_0(torch.cat([features, label_river_value], dim=1)), dim=-1)

            label_river_kicker_0 = river_kicker_0 if river_kicker_0 is not None else torch.argmax(kicker_river_0 , dim=1)
            label_river_kicker_0_enc = torch.nn.functional.one_hot(label_river_kicker_0.clamp(0), num_classes=13).float()
            label_river_kicker_0_enc[label_river_kicker_0==-2] = 0
            kicker_river_1 = torch.log_softmax(
                self.head_river_kicker_1(torch.cat([features, label_river_kicker_0_enc], dim=1)), dim=-1)

            label_river_kicker_1 = river_kicker_1 if river_kicker_1 is not None else torch.argmax(kicker_river_1 , dim=1)
            label_river_kicker_1_enc = torch.nn.functional.one_hot(label_river_kicker_1.clamp(0), num_classes=13).float()
            label_river_kicker_1_enc[label_river_kicker_1==-2] = 0
            kicker_river_2 = torch.log_softmax(
                self.head_river_kicker_2(torch.cat([features, label_river_kicker_1_enc], dim=1)), dim=-1)

            label_river_kicker_2 = river_kicker_2 if river_kicker_2 is not None else torch.argmax(kicker_river_2 , dim=1)
            label_river_kicker_2_enc = torch.nn.functional.one_hot(label_river_kicker_2.clamp(0), num_classes=13).float()
            label_river_kicker_2_enc[label_river_kicker_2==-2] = 0
            kicker_river_3 = torch.log_softmax(
                self.head_river_kicker_3(torch.cat([features, label_river_kicker_2_enc], dim=1)), dim=-1)

            label_river_kicker_3 = river_kicker_3 if river_kicker_3 is not None else torch.argmax(kicker_river_3 , dim=1)
            label_river_kicker_3_enc = torch.nn.functional.one_hot(label_river_kicker_3.clamp(0), num_classes=11).float()
            label_river_kicker_3_enc[label_river_kicker_3==-2] = 0
            kicker_river_4 = torch.log_softmax(
                self.head_river_kicker_4(torch.cat([features, label_river_kicker_3_enc], dim=1)), dim=-1)

            predictions.update(
                {
                    "log_probs_river_kicker_0": kicker_river_0,
                    "log_probs_river_kicker_1": kicker_river_1,
                    "log_probs_river_kicker_2": kicker_river_2,
                    "log_probs_river_kicker_3": kicker_river_3,
                    "log_probs_river_kicker_4": kicker_river_4,
                })

            # board kickers
            label_river_board_value = river_strength_board if river_strength_board is not None else torch.argmax(
                predictions["log_probs_board_strength_river"], dim=1)
            label_river_board_value = torch.nn.functional.one_hot(label_river_board_value.clamp(min=0), num_classes=9).float()
            kicker_river_board_0 = torch.log_softmax(
                self.head_board_river_kicker_0(torch.cat([features, label_river_board_value], dim=1)), dim=-1)

            label_river_board_kicker_0 = river_kicker_board_0 if river_kicker_board_0 is not None else torch.argmax(kicker_river_board_0, dim=1)
            label_river_board_kicker_0_enc = torch.nn.functional.one_hot(label_river_board_kicker_0.clamp(0),
                                                                   num_classes=13).float()
            label_river_board_kicker_0_enc[label_river_board_kicker_0 == -2] = 0
            kicker_river_board_1 = torch.log_softmax(
                self.head_board_river_kicker_1(torch.cat([features, label_river_board_kicker_0_enc], dim=1)), dim=-1)

            label_river_board_kicker_1 = river_kicker_board_1 if river_kicker_board_1 is not None else torch.argmax(kicker_river_board_1, dim=1)
            label_river_board_kicker_1_enc = torch.nn.functional.one_hot(label_river_board_kicker_1.clamp(0),
                                                                   num_classes=13).float()
            label_river_board_kicker_1_enc[label_river_board_kicker_1 == -2] = 0
            kicker_river_board_2 = torch.log_softmax(
                self.head_board_river_kicker_2(torch.cat([features, label_river_board_kicker_1_enc], dim=1)), dim=-1)

            label_river_board_kicker_2 = river_kicker_board_2 if river_kicker_board_2 is not None else torch.argmax(kicker_river_board_2, dim=1)
            label_river_board_kicker_2_enc = torch.nn.functional.one_hot(label_river_board_kicker_2.clamp(0),
                                                                   num_classes=13).float()
            label_river_board_kicker_2_enc[label_river_board_kicker_2 == -2] = 0
            kicker_river_board_3 = torch.log_softmax(
                self.head_board_river_kicker_3(torch.cat([features, label_river_board_kicker_2_enc], dim=1)), dim=-1)

            label_river_board_kicker_3 = river_kicker_board_3 if river_kicker_board_3 is not None else torch.argmax(kicker_river_board_3, dim=1)
            label_river_board_kicker_3_enc = torch.nn.functional.one_hot(label_river_board_kicker_3.clamp(0),
                                                                   num_classes=11).float()
            label_river_board_kicker_3_enc[label_river_board_kicker_3 == -2] = 0
            kicker_river_board_4 = torch.log_softmax(
                self.head_board_river_kicker_4(torch.cat([features, label_river_board_kicker_3_enc], dim=1)), dim=-1)

            predictions.update(
                {
                    "log_probs_river_board_kicker_0": kicker_river_board_0,
                    "log_probs_river_board_kicker_1": kicker_river_board_1,
                    "log_probs_river_board_kicker_2": kicker_river_board_2,
                    "log_probs_river_board_kicker_3": kicker_river_board_3,
                    "log_probs_river_board_kicker_4": kicker_river_board_4,
                }
            )
            predictions["log_probs_outcome_river"] = torch.log_softmax(self.head_outcome_probs_river(features), dim=-1)
        return features, predictions



embedding_dim = 8  # Size of individual card embeddings
if __name__ == "__main__":

    # Example usage
    feature_dim = 256  # Size of output feature vectors
    deep_layer_dims = (512, 2048, 2048, 2048)
    model = PokerHandEmbedding(embedding_dim, feature_dim, deep_layer_dims)

    # Example input: batch of 32 hands
    preflop = torch.randint(0, 52, (32, 2))  # 2 preflop cards
    flop = torch.randint(0, 52, (32, 3))  # 3 flop cards
    turn = torch.randint(0, 52, (32, 1))  # 1 turn card
    river = torch.randint(0, 52, (32, 1))  # 1 river card

    # Get feature vectors and probabilities
    preflop_features, preflop_probs = model(preflop)
    final_features, final_probs = model(preflop, flop, turn, river)

    print(preflop_features.shape)  # torch.Size([32, 128])
    print(preflop_probs.shape)  # torch.Size([32, 3])
    print(final_features.shape)  # torch.Size([32, 128])
    print(final_probs.shape)  # torch.Size([32, 3])
