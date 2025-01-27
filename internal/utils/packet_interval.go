package utils

import "github.com/shravan9912/mpquic_ml_a2c/internal/protocol"

// PacketInterval is an interval from one PacketNumber to the other
// +gen linkedlist
type PacketInterval struct {
	Start protocol.PacketNumber
	End   protocol.PacketNumber
}
