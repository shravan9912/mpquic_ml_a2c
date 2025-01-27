package constants

const (
	SCHEDULER_ROUND_ROBIN = "round_robin"
	SCHEDULER_SCH_SHRAVAN = "sch_shravan"
	SCHEDULER_LOW_LATENCY = "low_latency"
	SCHEDULER_RANDOM      = "random"
	SCHEDULER_LOW_BANDIT  = "low_bandit"
	SCHEDULER_PEEKABOO    = "peekaboo"
	SCHEDULER_ECF         = "ecf"
	SCHEDULER_BLEST       = "blest"
	SCHEDULER_AGENT         = "dqnAgent"
	SCHEDULER_FIRST_PATH  = "first_path"
	SCHEDULER_NEURAL_NET  = "neural_net"
	SCHEDULER_DEAR = "dear"
	SCHEDULER_MAC2 = "dearmac2"
	SCHEDULER_MAC3 = "dearmac3"
	SCHEDULER_CAP = "dearcap"
	SCHEDULER_PACE = "PACE"
	SCHEDULER_KUMAR = "kumar"
	SCHEDULER_PUMAR = "pumar"
	SCHEDULER_A2C = "a2c"
	SCHEDULER_PPO = "ppo"


	// Directory Names
	DEFAULT_TRAINING_DIR        = "online_training"
	DEFAULT_CLIENT_TRAINING_DIR = "client_online_training"
	DEFAULT_MODEL_OUTPUT_DIR    = "client_model_output"

	// File names
	TRAINING_FILE_NAME = "train.txt"
)

