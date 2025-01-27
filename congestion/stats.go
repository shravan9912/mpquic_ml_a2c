package congestion

import "github.com/shravan9912/mpquic_ml_a2c/internal/protocol"

type connectionStats struct {
	slowstartPacketsLost protocol.PacketNumber
	slowstartBytesLost   protocol.ByteCount
}
