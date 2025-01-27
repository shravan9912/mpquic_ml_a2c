package quic

import (
	"fmt"
	"math"
	"math/rand"
	"encoding/csv"
	"encoding/gob"
	"os"
	"strconv"
	"time"

	"github.com/Workiva/go-datastructures/queue"

	"gonum.org/v1/gonum/mat"

	deep "github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"

	wr "github.com/mroth/weightedrand"

	"github.com/shravan9912/gorl/agents"
	"github.com/shravan9912/gorl/types"
	"github.com/shravan9912/mpquic_ml_a2c/ackhandler"
	"github.com/shravan9912/mpquic_ml_a2c/constants"
	"github.com/shravan9912/mpquic_ml_a2c/internal/protocol"
	"github.com/shravan9912/mpquic_ml_a2c/internal/utils"
	"github.com/shravan9912/mpquic_ml_a2c/internal/wire"
	"github.com/shravan9912/mpquic_ml_a2c/util"
)

const banditAlpha = 0.75
const banditDimension = 6

type scheduler struct {
	// XXX Currently round-robin based, inspired from MPTCP scheduler
	quotas map[protocol.PathID]uint
	// Selected scheduler
	SchedulerName string
	// Is training?
	Training bool
	// Training Agent
	TrainingAgent agents.TrainingAgent
	// Normal Agent
	Agent agents.Agent

	// Cached state for training
	cachedState  types.Vector
	cachedPathID protocol.PathID

	AllowedCongestion int

	// async updated reward
	record        uint64
	episoderecord uint64
	statevector   [6000]types.Vector
	packetvector  [6000]uint64
	//rewardvector [6000]types.Output
	actionvector   [6000]int
	recordDuration [6000]types.Output
	lastfiretime   time.Time
	zz             [6000]time.Time
	waiting        uint64

	// linUCB
	fe           uint64
	se           uint64
	MAaF         [banditDimension][banditDimension]float64
	MAaS         [banditDimension][banditDimension]float64
	MbaF         [banditDimension]float64
	MbaS         [banditDimension]float64
	featureone   [6000]float64
	featuretwo   [6000]float64
	featurethree [6000]float64
	featurefour  [6000]float64
	featurefive  [6000]float64
	featuresix   [6000]float64
	// Retrans cache
	retrans map[protocol.PathID]uint64

	// Write experiences
	DumpExp   bool
	DumpPath  string
	dumpAgent experienceAgent

	// Project Home Directory
	projectHomeDir string

	// Neural Net Directory
	OnlineTrainingFile string
	ModelOutputDir     string

	WriteHeaderColumn bool

	// pathID Queue for Round Robin
	pathQueue queue.Queue

	// dear
	actor *deep.Neural
	critic *deep.Neural
	trainer *training.OnlineTrainer
	pval float64
	gamma float64
	rewriteWeights bool

	// actorm2d-Criticab
	actorm2d *deep.Neural
	criticm2a *deep.Neural
	criticm2b *deep.Neural
	trainerm2ab *training.OnlineTrainer
	pvalm2a float64
	pvalm2b float64
	gammam2ab float64
	rm21 float64
	rm22 float64
	rewriteWeightsm2ab bool
	ukkm2 int64
	// Actord-Criticabc
	actorm3d *deep.Neural
	criticm3a *deep.Neural
	criticm3b *deep.Neural
	criticm3c *deep.Neural
	trainerm3ab *training.OnlineTrainer
	pvalm3a float64
	pvalm3b float64
	gammam3ab float64
	r1m3 float64
	r2m3 float64
	rewriteWeightsm3abc bool
	ukkm3 int64
	// CAP
	actorcapd *deep.Neural
	criticcapa *deep.Neural
	criticcapb *deep.Neural
	trainercapab *training.OnlineTrainer
	pvalcapa float64
	pvalcapb float64
	gammacapab float64
	r1cap float64
	r2cap float64
	rewriteWeightscapab bool
	ukkcap int64
	
	// PACE
	actorpa *deep.Neural
	actorpb *deep.Neural
	criticpa *deep.Neural
	criticpb *deep.Neural
	trainerpab *training.OnlineTrainer
	pvalpa float64
	pvalpb float64
	gammapab float64
	rp1 float64
	rp2 float64
	rewriteWeightspab bool
	ukkm2p int64
        x int64
        k int64
        aa int64
         
	
}
//for a2c helpers start

// Hyperparameters
const (
	learningRate = 0.001
	gamma        = 0.99 // Discount factor for rewards
	stateDim     = 6    // RTT, CWND, inflight for 2 paths
	actionDim    = 2    // Path1 or Path2
)

// ActorCritic represents the model
type ActorCritic struct {
	weights *mat.Dense // Weights for the policy
	values  *mat.Dense // Weights for the value function
}
var model *ActorCritic // Declare model outside the if bloc
//for a2c helpers end
type queuePathIdItem struct {
	pathId protocol.PathID
	path   *path
}
// Agent represents a DRL agent with specific weights for the reward function.
 type Agent struct {
	             Name       string
	             wRTT       float64
	             wCWND      float64
	             wInflight  float64
  }
 // Path represents the parameters of a network path.


var (
	               PPM_wRTT      = -1.0
	               PPM_wCWND     = 0.5
	               PPM_wInflight = 0.5
                    )
func (sch *scheduler) setup() {
	sch.projectHomeDir = os.Getenv(constants.PROJECT_HOME_DIR)
	if sch.projectHomeDir == "" {
		panic("`PROJECT_HOME_DIR` Env variable was not provided, this is needed for training")
	}
	sch.quotas = make(map[protocol.PathID]uint)
	sch.retrans = make(map[protocol.PathID]uint64)
	sch.waiting = 0

	sch.WriteHeaderColumn = true

	//Read lin to buffer
	linFileName := sch.projectHomeDir + "/sch_out/lin"
	file, err := os.OpenFile(linFileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		panic(err)
	}

	for i := 0; i < banditDimension; i++ {
		for j := 0; j < banditDimension; j++ {
			fmt.Fscanln(file, &sch.MAaF[i][j])
		}
	}
	for i := 0; i < banditDimension; i++ {
		for j := 0; j < banditDimension; j++ {
			fmt.Fscanln(file, &sch.MAaS[i][j])
		}
	}
	for i := 0; i < banditDimension; i++ {
		fmt.Fscanln(file, &sch.MbaF[i])
	}
	for i := 0; i < banditDimension; i++ {
		fmt.Fscanln(file, &sch.MbaS[i])
	}
	file.Close()

	//TODO: expose to config
	sch.DumpPath = "/tmp/"
	sch.dumpAgent.Setup()

	sch.cachedState = types.Vector{-1, -1}
	if sch.SchedulerName == "dqnAgent" {
		if sch.Training {
			sch.TrainingAgent = GetTrainingAgent("", "", "", 0.)
		} else {
			sch.Agent = GetAgent("", "")
		}
	}

	if sch.SchedulerName == "dear" {
		// change values
		fmt.Println("Initialized actor and critic")
		optimizer := training.NewAdam(0.001, 0.9, 0.999, 1e-8)
		sch.trainer = training.NewTrainer(optimizer, 0)
		sch.gamma = 0.99
		// Actor network
		sch.actor = deep.NewNeural(&deep.Config{
			/* Input dimensionality */
			Inputs: 6,
			/* Two hidden layers consisting of two neurons each, and a single output */
			Layout: []int{8, 4, 2},
			/* Activation functions: Sigmoid, Tanh, ReLU, Linear */
			Activation: deep.ActivationReLU,
			/* Determines output layer activation & loss function */
			Mode: deep.ModeMultiClass,
			/* Weight initializers: {deep.NewNormal(μ, σ), deep.NewUniform(μ, σ)} */
			Weight: deep.NewUniform(1.0, 0.0),
			/* Apply bias */
			Loss: 4,
			Bias: false,
		})
		// Critic network
		sch.critic = deep.NewNeural(&deep.Config{
			Inputs: 6,
			Layout: []int{8, 4, 1},
			Activation: deep.ActivationReLU,
			Mode: deep.ModeDefault,
			Weight: deep.NewUniform(1.0, 0.0),
			Loss: 5,
			Bias: false,
		})
		if !sch.rewriteWeights {
			fmt.Println("Loading actor(policy) weights")
			actorFile, err := os.Open(sch.projectHomeDir+"/sch_out/actorweights")
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			for i, l := range sch.actor.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(actorFile, "%f\n", &sch.actor.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			actorFile.Close()
			criticFile, err := os.Open(sch.projectHomeDir+"/sch_out/actorweights")
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			for i, l := range sch.critic.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(criticFile, "%f\n", &sch.critic.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			criticFile.Close()
		}
	}
//a2c config start

if sch.SchedulerName == "a2c" {

// Hyperparameters
const (
	learningRate = 0.001
	gamma        = 0.99 // Discount factor for rewards
	stateDim     = 6    // RTT, CWND, inflight for 2 paths
	actionDim    = 2    // Path1 or Path2
)
	weights := mat.NewDense(actionDim, stateDim, randomArray(actionDim*stateDim))
	values := mat.NewDense(1, stateDim, randomArray(stateDim))
	model := &ActorCritic{weights: weights, values: values}
fmt.Println("Model created:", model) // Use the model to avoid 'not used' error
fmt.Println("CALLING SETUP AAAAAAAAAAAAAAAAAAAAAA")

  sch.aa=0
// CreateModel initializes the ActorCritic model
//func CreateModel() *ActorCritic {

}

//a2c config end
	
if sch.SchedulerName == "kumar" {
                //kumar start
                fmt.Println("CALLING SETUP KUMAR")
                // Agent represents a DRL agent with specific weights for the reward function.
                type Agent struct {
	             Name       string
	             wRTT       float64
	             wCWND      float64
	             wInflight  float64
                 }
                sch.x=0
                
                //kumarends



		// change values

	}	
	
if sch.SchedulerName == "pumar" {
                fmt.Println("CALLING SETUP PUMAR")
                //kumar start
                // Agent represents a DRL agent with specific weights for the reward function.
                type Agent struct {
	             Name       string
	             wRTT       float64
	             wCWND      float64
	             wInflight  float64
                 }
                sch.k=0
                
                //kumarends



		// change values

	}	
		
	
// dearmac2
	if sch.SchedulerName == "dearmac2" {
		// change values
		fmt.Println("Initialized actorm2d and criticm2ab")
		optimizer := training.NewAdam(0.001, 0.9, 0.999, 1e-8)
		sch.trainer = training.NewTrainer(optimizer, 0)
		sch.gammam2ab = 0.99
		// Actor network
		sch.actorm2d = deep.NewNeural(&deep.Config{
			/* Input dimensionality */
			Inputs: 6,
			/* Two hidden layers consisting of two neurons each, and a single output */
			Layout: []int{8, 4, 2},
			/* Activation functions: Sigmoid, Tanh, ReLU, Linear */
			Activation: deep.ActivationReLU,
			/* Determines output layer activation & loss function */
			Mode: deep.ModeMultiClass,
			/* Weight initializers: {deep.NewNormal(μ, σ), deep.NewUniform(μ, σ)} */
			Weight: deep.NewUniform(1.0, 0.0),
			/* Apply bias */
			Loss: 4,
			Bias: false,
		})
		// Critic network
		sch.criticm2a = deep.NewNeural(&deep.Config{
			Inputs: 6,
			Layout: []int{8, 4, 1},
			Activation: deep.ActivationReLU,
			Mode: deep.ModeDefault,
			Weight: deep.NewUniform(1.0, 0.0),
			Loss: 5,
			Bias: false,
		})
		sch.criticm2b = deep.NewNeural(&deep.Config{
			Inputs: 6,
			Layout: []int{8, 4, 1},
			Activation: deep.ActivationReLU,
			Mode: deep.ModeDefault,
			Weight: deep.NewUniform(1.0, 0.0),
			Loss: 5,
			Bias: false,
		})

		if !sch.rewriteWeightsm2ab {
			fmt.Println("Loading actor(policy) weights")
			actorm2dFile, err := os.Open(sch.projectHomeDir+"/sch_out/actorm2dweights")
			fmt.Println("xxxx")			
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			fmt.Println("xxxx1")
			for i, l := range sch.actorm2d.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(actorm2dFile, "%f\n", &sch.actorm2d.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			fmt.Println("xxxx2")
			actorm2dFile.Close()
			criticm2aFile, err := os.Open(sch.projectHomeDir+"/sch_out/criticm2aweights")
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			for i, l := range sch.criticm2a.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(criticm2aFile, "%f\n", &sch.criticm2a.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			criticm2aFile.Close()
			criticm2bFile, err := os.Open(sch.projectHomeDir+"/sch_out/criticm2bweights")
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			for i, l := range sch.criticm2b.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(criticm2bFile, "%f\n", &sch.criticm2b.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			fmt.Println("xxxx8")
			criticm2bFile.Close()
			fmt.Println("xxxx9")
		}
	}	
// dearmac3
	if sch.SchedulerName == "dearmac3" {
		// change values
		fmt.Println("Initialized actorm3d and criticab")
		optimizer := training.NewAdam(0.001, 0.9, 0.999, 1e-8)
		sch.trainer = training.NewTrainer(optimizer, 0)
		sch.gammam3ab = 0.99
		// Actor network
		sch.actorm3d = deep.NewNeural(&deep.Config{
			/* Input dimensionality */
			Inputs: 6,
			/* Two hidden layers consisting of two neurons each, and a single output */
			Layout: []int{8, 4, 2},
			/* Activation functions: Sigmoid, Tanh, ReLU, Linear */
			Activation: deep.ActivationReLU,
			/* Determines output layer activation & loss function */
			Mode: deep.ModeMultiClass,
			/* Weight initializers: {deep.NewNormal(μ, σ), deep.NewUniform(μ, σ)} */
			Weight: deep.NewUniform(1.0, 0.0),
			/* Apply bias */
			Loss: 4,
			Bias: false,
		})
		// Critic network
		sch.criticm3a = deep.NewNeural(&deep.Config{
			Inputs: 6,
			Layout: []int{8, 4, 1},
			Activation: deep.ActivationReLU,
			Mode: deep.ModeDefault,
			Weight: deep.NewUniform(1.0, 0.0),
			Loss: 5,
			Bias: false,
		})
		sch.criticm3b = deep.NewNeural(&deep.Config{
			Inputs: 6,
			Layout: []int{8, 4, 1},
			Activation: deep.ActivationReLU,
			Mode: deep.ModeDefault,
			Weight: deep.NewUniform(1.0, 0.0),
			Loss: 5,
			Bias: false,
		})
                sch.criticm3c = deep.NewNeural(&deep.Config{
			Inputs: 6,
			Layout: []int{8, 4, 1},
			Activation: deep.ActivationReLU,
			Mode: deep.ModeDefault,
			Weight: deep.NewUniform(1.0, 0.0),
			Loss: 5,
			Bias: false,
		})

		if !sch.rewriteWeightsm3abc {
			fmt.Println("Loading actor(policy) weights")
			actordFile, err := os.Open(sch.projectHomeDir+"/sch_out/actorm3dweights")
			fmt.Println("xxxx")			
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			fmt.Println("xxxx1")
			for i, l := range sch.actorm3d.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(actordFile, "%f\n", &sch.actorm3d.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			fmt.Println("xxxx2")
			actordFile.Close()
			criticaFile, err := os.Open(sch.projectHomeDir+"/sch_out/criticm3aweights")
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			for i, l := range sch.criticm3a.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(criticaFile, "%f\n", &sch.criticm3a.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			criticaFile.Close()
			criticbFile, err := os.Open(sch.projectHomeDir+"/sch_out/criticm3bweights")
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			for i, l := range sch.criticm3b.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(criticbFile, "%f\n", &sch.criticm3b.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			fmt.Println("xxxx8")
			criticbFile.Close()
			fmt.Println("xxxx9")
			criticcFile, err := os.Open(sch.projectHomeDir+"/sch_out/criticm3cweights")
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			for i, l := range sch.criticm3c.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(criticcFile, "%f\n", &sch.criticm3c.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}

			criticcFile.Close()
		}
	}
// dearcap
	if sch.SchedulerName == "dearcap" {
		// change values
		fmt.Println("Initialized actor and critic per path")
		optimizer := training.NewAdam(0.001, 0.9, 0.999, 1e-8)
		sch.trainer = training.NewTrainer(optimizer, 0)
		sch.gammacapab = 0.99
		// Actor network
		sch.actorcapd = deep.NewNeural(&deep.Config{
			/* Input dimensionality */
			Inputs: 6,
			/* Two hidden layers consisting of two neurons each, and a single output */
			Layout: []int{8, 4, 2},
			/* Activation functions: Sigmoid, Tanh, ReLU, Linear */
			Activation: deep.ActivationReLU,
			/* Determines output layer activation & loss function */
			Mode: deep.ModeMultiClass,
			/* Weight initializers: {deep.NewNormal(μ, σ), deep.NewUniform(μ, σ)} */
			Weight: deep.NewUniform(1.0, 0.0),
			/* Apply bias */
			Loss: 4,
			Bias: false,
		})
		// Critic network
		sch.criticcapa = deep.NewNeural(&deep.Config{
			Inputs: 6,
			Layout: []int{8, 4, 1},
			Activation: deep.ActivationReLU,
			Mode: deep.ModeDefault,
			Weight: deep.NewUniform(1.0, 0.0),
			Loss: 5,
			Bias: false,
		})
		sch.criticcapb = deep.NewNeural(&deep.Config{
			Inputs: 6,
			Layout: []int{8, 4, 1},
			Activation: deep.ActivationReLU,
			Mode: deep.ModeDefault,
			Weight: deep.NewUniform(1.0, 0.0),
			Loss: 5,
			Bias: false,
		})

		if !sch.rewriteWeightscapab {
			fmt.Println("Loading actor(policy) weights")
			actorcapdFile, err := os.Open(sch.projectHomeDir+"/sch_out/actorcapdweights")
			fmt.Println("xxxx")			
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			fmt.Println("xxxx1")
			for i, l := range sch.actorcapd.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(actorcapdFile, "%f\n", &sch.actorcapd.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			fmt.Println("xxxx2")
			actorcapdFile.Close()
			criticcapaFile, err := os.Open(sch.projectHomeDir+"/sch_out/criticcapaweights")
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			for i, l := range sch.criticcapa.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(criticcapaFile, "%f\n", &sch.criticcapa.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			criticcapaFile.Close()
			criticcapbFile, err := os.Open(sch.projectHomeDir+"/sch_out/criticcapbweights")
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			for i, l := range sch.criticcapb.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(criticcapbFile, "%f\n", &sch.criticcapb.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			fmt.Println("xxxx8")
			criticcapbFile.Close()
			fmt.Println("xxxx9")
		}
	}	
	


		
// PACE
	if sch.SchedulerName == "PACE" {
		// change values
		fmt.Println("Initialized actorpab and criticpab")
		optimizer := training.NewAdam(0.001, 0.9, 0.999, 1e-8)
		sch.trainer = training.NewTrainer(optimizer, 0)
		sch.gammam2ab = 0.99
		// Actorpa network
		sch.actorpa = deep.NewNeural(&deep.Config{
			/* Input dimensionality */
			Inputs: 6,
			/* Two hidden layers consisting of two neurons each, and a single output */
			Layout: []int{8, 4, 2},
			/* Activation functions: Sigmoid, Tanh, ReLU, Linear */
			Activation: deep.ActivationReLU,
			/* Determines output layer activation & loss function */
			Mode: deep.ModeMultiClass,
			/* Weight initializers: {deep.NewNormal(μ, σ), deep.NewUniform(μ, σ)} */
			Weight: deep.NewUniform(1.0, 0.0),
			/* Apply bias */
			Loss: 4,
			Bias: false,
		})
		// Actorpb network
		sch.actorpb = deep.NewNeural(&deep.Config{
			/* Input dimensionality */
			Inputs: 6,
			/* Two hidden layers consisting of two neurons each, and a single output */
			Layout: []int{8, 4, 2},
			/* Activation functions: Sigmoid, Tanh, ReLU, Linear */
			Activation: deep.ActivationReLU,
			/* Determines output layer activation & loss function */
			Mode: deep.ModeMultiClass,
			/* Weight initializers: {deep.NewNormal(μ, σ), deep.NewUniform(μ, σ)} */
			Weight: deep.NewUniform(1.0, 0.0),
			/* Apply bias */
			Loss: 4,
			Bias: false,
		})
		
		// Critic network
		sch.criticpa = deep.NewNeural(&deep.Config{
			Inputs: 6,
			Layout: []int{8, 4, 1},
			Activation: deep.ActivationReLU,
			Mode: deep.ModeDefault,
			Weight: deep.NewUniform(1.0, 0.0),
			Loss: 5,
			Bias: false,
		})
		sch.criticpb = deep.NewNeural(&deep.Config{
			Inputs: 6,
			Layout: []int{8, 4, 1},
			Activation: deep.ActivationReLU,
			Mode: deep.ModeDefault,
			Weight: deep.NewUniform(1.0, 0.0),
			Loss: 5,
			Bias: false,
		})

		if !sch.rewriteWeightspab {
			fmt.Println("Loading actorpa(policy) weights")
			actorpaFile, err := os.Open(sch.projectHomeDir+"/sch_out/actorpaweights")
			fmt.Println("xxxx")			
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			fmt.Println("xxxx1")
			for i, l := range sch.actorpa.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(actorpaFile, "%f\n", &sch.actorpa.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			fmt.Println("xxxx2")
			actorpaFile.Close()
			fmt.Println("Loading actorpb(policy) weights")
			actorpbFile, err := os.Open(sch.projectHomeDir+"/sch_out/actorpbweights")
			fmt.Println("xxxx")			
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			fmt.Println("xxxx1")
			for i, l := range sch.actorpb.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(actorpbFile, "%f\n", &sch.actorpb.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			fmt.Println("xxxx2")
			actorpbFile.Close()
			criticpaFile, err := os.Open(sch.projectHomeDir+"/sch_out/criticpaweights")
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			for i, l := range sch.criticpa.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(criticpaFile, "%f\n", &sch.criticpa.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			criticpaFile.Close()
			criticpbFile, err := os.Open(sch.projectHomeDir+"/sch_out/criticpbweights")
			if(err != nil){
				fmt.Println("Error in opening file")
				fmt.Println(err)
			}
			for i, l := range sch.criticpb.Layers {
				for j := range l.Neurons {
					for k := range l.Neurons[j].In {
						fmt.Fscanf(criticpbFile, "%f\n", &sch.criticpb.Layers[i].Neurons[j].In[k].Weight)
					}
				}
			}
			fmt.Println("xxxx8")
			criticpbFile.Close()
			fmt.Println("xxxx9")
		}
	}
	
}

func (sch *scheduler) getRetransmission(s *session) (hasRetransmission bool, retransmitPacket *ackhandler.Packet, pth *path) {
	// check for retransmissions first
	for {
		// TODO add ability to reinject on another path
		// XXX We need to check on ALL paths if any packet should be first retransmitted
		s.pathsLock.RLock()
	retransmitLoop:
		for _, pthTmp := range s.paths {
			retransmitPacket = pthTmp.sentPacketHandler.DequeuePacketForRetransmission()
			if retransmitPacket != nil {
				pth = pthTmp
				break retransmitLoop
			}
		}
		s.pathsLock.RUnlock()
		if retransmitPacket == nil {
			break
		}
		hasRetransmission = true

		if retransmitPacket.EncryptionLevel != protocol.EncryptionForwardSecure {
			if s.handshakeComplete {
				// Don't retransmit handshake packets when the handshake is complete
				continue
			}
			utils.Debugf("\tDequeueing handshake retransmission for packet 0x%x", retransmitPacket.PacketNumber)
			return
		}
		utils.Debugf("\tDequeueing retransmission of packet 0x%x from path %d", retransmitPacket.PacketNumber, pth.pathID)
		// resend the frames that were in the packet
		for _, frame := range retransmitPacket.GetFramesForRetransmission() {
			switch f := frame.(type) {
			case *wire.StreamFrame:
				s.streamFramer.AddFrameForRetransmission(f)
			case *wire.WindowUpdateFrame:
				// only retransmit WindowUpdates if the stream is not yet closed and the we haven't sent another WindowUpdate with a higher ByteOffset for the stream
				// XXX Should it be adapted to multiple paths?
				currentOffset, err := s.flowControlManager.GetReceiveWindow(f.StreamID)
				if err == nil && f.ByteOffset >= currentOffset {
					s.packer.QueueControlFrame(f, pth)
				}
			case *wire.PathsFrame:
				// Schedule a new PATHS frame to send
				s.schedulePathsFrame()
			default:
				s.packer.QueueControlFrame(frame, pth)
			}
		}
	}
	return
}

func (sch *scheduler) selectPathRoundRobin(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	if sch.quotas == nil {
		sch.setup()
	}

	// Log Path Id w/ Interface Name
	//for pathId, pth := range s.paths {
	//	fmt.Printf("Path Id: %d, Local Addr: %s, Remote Addr: %s \t", pathId, pth.conn.LocalAddr(), pth.conn.RemoteAddr())
	//}

	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}

	if sch.pathQueue.Empty() {
		for pathId, pth := range s.paths {
			err := sch.pathQueue.Put(queuePathIdItem{pathId: pathId, path: pth})
			if err != nil {
				fmt.Println("Err Inserting in Queue, Error: ", err.Error())
			}
		}
	} else if int64(len(s.paths)) != sch.pathQueue.Len() {
		sch.pathQueue.Get(sch.pathQueue.Len())
		for pathId, pth := range s.paths {
			err := sch.pathQueue.Put(queuePathIdItem{pathId: pathId, path: pth})
			if err != nil {
				fmt.Println("Err Inserting in Queue, Error: ", err.Error())
			}
		}
	}

pathLoop:
	for pathID, pth := range s.paths {
		pathIdFromQueue, _ := sch.pathQueue.Peek()
		pathIdObj, ok := pathIdFromQueue.(queuePathIdItem)
		if !ok {
			panic("Invalid Interface Type Chosen")
		}

		// Don't block path usage if we retransmit, even on another path
		// If this path is potentially failed, do no consider it for sending
		// XXX Prevent using initial pathID if multiple paths
		if (!hasRetransmission && !pth.SendingAllowed()) || pth.potentiallyFailed.Get() || pathID == protocol.InitialPathID {
			if pathIdObj.pathId == pathID {
				_, _ = sch.pathQueue.Get(1)
				_ = sch.pathQueue.Put(pathIdObj)
			}
			continue pathLoop
		}

		if pathIdObj.pathId == pathID {
			_, _ = sch.pathQueue.Get(1)
			_ = sch.pathQueue.Put(pathIdObj)
			return pth
		}
	}
	return nil

}

// dear scheduler
func (sch *scheduler) selectPathdear(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
        //fmt.Println(time.Since(s.sessionCreationTime))
	if sch.actor == nil || sch.critic == nil {
		fmt.Println("CALLING SETUP")
		fmt.Println(sch.SchedulerName)
		sch.setup()
	}

	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}
	
	state := []float64{}
	sRTT := make(map[protocol.PathID]time.Duration)

	//Check for available paths
	var availablePaths []protocol.PathID
	for pathID, path := range s.paths {
		if path.sentPacketHandler.SendingAllowed() && pathID != protocol.InitialPathID {
			availablePaths = append(availablePaths, pathID)
			//fmt.Println("Available: ", pathID)
		}
		if pathID != protocol.InitialPathID {
			RTTs := path.rttStats.LatestRTT()
			CWND := path.sentPacketHandler.GetCongestionWindow()
			Inf := path.sentPacketHandler.GetBytesInFlight()
			/*BSend, err := s.flowControlManager.SendWindowSize(protocol.StreamID(5))
			if err != nil{
				fmt.Print("error in getting sendwin size: ")
				//fmt.Print(s.flowControlManager.streamFlowController)
				fmt.Println(err)
			}*/
			state = append(state, float64(RTTs.Microseconds())/1000)
			state = append(state, float64(CWND)/1000)
			state = append(state, float64(Inf))
		}
	}
	
	//fmt.Println("pathids len: ", len(pathids))
	if len(availablePaths) == 0 {
		if s.paths[protocol.InitialPathID].SendingAllowed() || hasRetransmission {
			return s.paths[protocol.InitialPathID]
		} else {
			return nil
		}
	}

	if len(availablePaths) == 1 {
		// TODO: Find a way to wait/transmit	 when only one path is available
		return s.paths[availablePaths[0]]
	}

	pi := []float64{}

	if sch.Training == false {
		//fmt.Print("state: ")
		//fmt.Println(state)
		//fmt.Print("predicted: ")
		pi = sch.actor.Predict(state)
		//fmt.Println(pi)
	} else {
		/*
		if training:
		-> Get CWND, SWND, InFlight, RTT for both paths i.e. the state
		-> get action_probs & state_value from agent
		-> Get the new_state_value using the agent and find the value function
		-> Take action and calculate reward = PS/Tack for a given packet
		-> delta = reward + sch.gamma * new_state_value - state_value
		-> actor_loss = -log_prob(Actionprobs) * delta // A list of log action probabilities, multiplied by delta. Used as the loss function for the actor network
		-> critic_loss = delta^2
		-> Use the losses and update the actor and critic networks
		*/
		duration := time.Since(s.sessionCreationTime)
		var maxRTT time.Duration
		for pathID := range sRTT {
			if sRTT[pathID] > maxRTT {
				maxRTT = sRTT[pathID]
			}
		}

		if(sch.pval == 0){
			sch.pval = sch.critic.Predict(state)[0]
		}
		pi = sch.actor.Predict(state)
		//fmt.Print("Pi: ",pi)
		v := (sch.critic.Predict(state))[0]
		//fmt.Print("v: ",v)
		reward := RewardFinalGoodput(sch, s, duration, maxRTT)
		delta := float64(reward) + sch.gamma*v - sch.pval;
		//fmt.Println(pi, delta)
		sch.pval = v;
		actor_loss := make([]float64, len(pi))
		for i, x := range(pi){
			actor_loss[i] = -delta*math.Log(x)
		}
		//critic_loss := delta * delta
		//fmt.Print("calculated losses: ")
		//fmt.Println(actor_loss, critic_loss)

		actor_data := training.Examples{
			training.Example{state, []float64{delta, delta}},
		}
		sch.trainer.Train(sch.actor, actor_data, nil, 1)

		critic_data := training.Examples{
			training.Example{state, []float64{delta*sch.gamma}},
		}
		sch.trainer.Train(sch.critic, critic_data, nil, 1)
		/*fmt.Print("state: ")
		fmt.Println(state)
		fmt.Print("predicted: ")
		fmt.Println(pi)*/
	}

    //fmt.Println("Selecting path %d", selectedPath)

	rand.Seed(time.Now().UTC().UnixNano()) // always seed random!

	pathsToSelect := []wr.Choice{}
	for i, _ := range availablePaths{
		pathsToSelect = append(pathsToSelect, wr.NewChoice(i, uint(pi[i]*100)))
	}
    chooser, _ := wr.NewChooser(
		wr.NewChoice(0, uint(pi[0]*100)),
		wr.NewChoice(1, uint(pi[1]*100)),
	)
	// Choose from prob distribution 
    result := chooser.Pick().(int)
    //fmt.Println("selected choice: ", result)
        //duration: = time.Since(s.sessionCreationTime)
	// If both paths are available, decide the path based on the probabilities given by the policy
	//fmt.Println(s.paths[availablePaths[0]])
	return s.paths[availablePaths[result]]
}
//a2c required functions start

func CreateModel() *ActorCritic {
	weights := mat.NewDense(actionDim, stateDim, randomArray(actionDim*stateDim))
	values := mat.NewDense(1, stateDim, randomArray(stateDim))
	return &ActorCritic{weights: weights, values: values}
}
// ComputeReward calculates the reward based on the selected action
func ComputeReward(state *mat.VecDense, action int) float64 {
	rtt1, cwnd1, inflight1 := state.AtVec(0), state.AtVec(1), state.AtVec(2)
	rtt2, cwnd2, inflight2 := state.AtVec(3), state.AtVec(4), state.AtVec(5)

	if action == 0 { // Path1 selected
		return cwnd1 / (rtt1 + inflight1)
	}
	return cwnd2 / (rtt2 + inflight2) // Path2 selected
}

// SelectAction uses the model to choose the best path
func SelectAction(model *ActorCritic, state *mat.VecDense) int {
	// Compute logits: policy = weights * state
	logits := mat.NewVecDense(actionDim, nil)
	logits.MulVec(model.weights, state)

	// Compute softmax probabilities
	probs := softmax(logits.RawVector().Data)
	return argMax(probs)
}

// Softmax function to compute probabilities
func softmax(logits []float64) []float64 {
	maxLogit := max(logits)
	expSum := 0.0
	for i, val := range logits {
		logits[i] = math.Exp(val - maxLogit)
		expSum += logits[i]
	}
	for i := range logits {
		logits[i] /= expSum
	}
	return logits
}

// ArgMax function to get the index of the max value
func argMax(array []float64) int {
	maxIdx, maxVal := 0, array[0]
	for i, val := range array {
		if val > maxVal {
			maxIdx, maxVal = i, val
		}
	}
	return maxIdx
}

// Helper function to generate random weights
func randomArray(size int) []float64 {
	arr := make([]float64, size)
	for i := range arr {
		arr[i] = rand.Float64() * 0.1 // Small random weights
	}
	return arr
}

// Max function to find the maximum in a slice
func max(array []float64) float64 {
	maxVal := array[0]
	for _, val := range array {
		if val > maxVal {
			maxVal = val
		}
	}
	return maxVal
}

// Sample an action based on probabilities
func sampleAction(probs []float64) int {
	r := rand.Float64()
	cumProb := 0.0
	for i, prob := range probs {
		cumProb += prob
		if r <= cumProb {
			return i
		}
	}
	return len(probs) - 1
}

// Check for NaN or Inf in a value
func isInvalidNumber(x float64) bool {
	return math.IsNaN(x) || math.IsInf(x, 0)
}
func clip(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}
func normalizeVector(vec *mat.VecDense) {
	sum := 0.0
	for i := 0; i < vec.Len(); i++ {
		sum += vec.AtVec(i)
	}
	for i := 0; i < vec.Len(); i++ {
		vec.SetVec(i, vec.AtVec(i)/sum)
	}
}
func normalizeMatrix(matrix *mat.Dense) {
	rows, cols := matrix.Dims()
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < cols; j++ {
			sum += matrix.At(i, j) * matrix.At(i, j)
		}
		norm := math.Sqrt(sum)
		if norm > 0 {
			for j := 0; j < cols; j++ {
				matrix.Set(i, j, matrix.At(i, j)/norm)
			}
		}
	}
}
func isValidMatrix(matrix *mat.Dense) bool {
	rows, cols := matrix.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if isInvalidNumber(matrix.At(i, j)) {
				return false
			}
		}
	}
	return true
}
func isValidVector(vec *mat.VecDense) bool {
	for i := 0; i < vec.Len(); i++ {
		if isInvalidNumber(vec.AtVec(i)) {
			return false
		}
	}
	return true
}



// UpdateModel updates the model's weights based on rewards and predictions
/*func UpdateModel(model *ActorCritic, state *mat.VecDense, action int, reward, discountedReward float64) {
	// Compute the advantage: A = discountedReward - V(state)
	value := mat.Dot(model.values.RowView(0), state)
	advantage := discountedReward - value

	// Update policy weights (policy gradient)
	for i := 0; i < actionDim; i++ {
		for j := 0; j < stateDim; j++ {
			grad := 0.0
			if i == action {
				grad = advantage * state.AtVec(j)
			}
			model.weights.Set(i, j, model.weights.At(i, j)+learningRate*grad)
		}
	}

	// Update value weights (value loss)
	for j := 0; j < stateDim; j++ {
		grad := advantage * state.AtVec(j)
		model.values.Set(0, j, model.values.At(0, j)+learningRate*grad)
	}
}
func UpdateModel(model *ActorCritic, state *mat.VecDense, action int, reward, discountedReward float64) {
	// Normalize the state vector
	normalizeVector(state)

	// Compute the value
	value := mat.Dot(model.values.RowView(0), state)

	// Handle invalid value
	if isInvalidNumber(value) {
		fmt.Println("Error: Invalid value detected!")
		return
	}

	// Compute the advantage: A = discountedReward - V(state)
	advantage := discountedReward - value

	// Clip advantage to avoid instability
	advantage = clip(advantage, -10.0, 10.0)

	// Update policy weights (policy gradient)
	for i := 0; i < actionDim; i++ {
		for j := 0; j < stateDim; j++ {
			grad := 0.0
			if i == action {
				grad = advantage * state.AtVec(j)
			}
			grad = clip(grad, -1.0, 1.0) // Clip gradients
			model.weights.Set(i, j, model.weights.At(i, j)+learningRate*grad)
		}
	}

	// Update value weights (value loss)
	for j := 0; j < stateDim; j++ {
		grad := advantage * state.AtVec(j)
		grad = clip(grad, -1.0, 1.0) // Clip gradients
		model.values.Set(0, j, model.values.At(0, j)+learningRate*grad)
	}
}*/

func UpdateModel(model *ActorCritic, state *mat.VecDense, action int, discountedReward float64) {
	// Validate state
	if !isValidVector(state) {
		fmt.Println("Error: Invalid state vector!")
		return
	}
	normalizeVector(state)

	// Validate model values
	if !isValidMatrix(model.values) {
		fmt.Println("Error: Invalid model values!")
		return
	}

	// Clip discountedReward
	discountedReward = clip(discountedReward, -10.0, 10.0)

	// Compute value
	value := mat.Dot(model.values.RowView(0), state)
	if isInvalidNumber(value) {
		fmt.Println("Error: Invalid value detected!")
		return
	}

	// Compute advantage
	advantage := discountedReward - value
	advantage = clip(advantage, -10.0, 10.0)

	// Update policy weights
	for i := 0; i < actionDim; i++ {
		for j := 0; j < stateDim; j++ {
			grad := 0.0
			if i == action {
				grad = advantage * state.AtVec(j)
			}
			grad = clip(grad, -1.0, 1.0)
			model.weights.Set(i, j, model.weights.At(i, j)+learningRate*grad)
		}
	}

	// Update value weights
	for j := 0; j < stateDim; j++ {
		grad := advantage * state.AtVec(j)
		grad = clip(grad, -1.0, 1.0)
		model.values.Set(0, j, model.values.At(0, j)+learningRate*grad)
	}

	// Regularize weights and values
	normalizeMatrix(model.weights)
	normalizeMatrix(model.values)
}



// CalculateDiscountedRewards calculates the discounted reward for each time step
func CalculateDiscountedRewards(rewards []float64) []float64 {
	discountedRewards := make([]float64, len(rewards))
	runningSum := 0.0
	for t := len(rewards) - 1; t >= 0; t-- {
		runningSum = rewards[t] + gamma*runningSum
		discountedRewards[t] = runningSum
	}
	return discountedRewards
}

// SaveModel saves the policy and value weights to files
func SaveModel(model *ActorCritic, policyFile, valueFile string) error {
	// Save policy weights
	if err := saveMatrixToFile(model.weights, policyFile); err != nil {
		return err
	}

	// Save value weights
	if err := saveMatrixToFile(model.values, valueFile); err != nil {
		return err
	}

	return nil
}

// LoadModel loads the policy and value weights from files
func LoadModel(policyFile, valueFile string) (*ActorCritic, error) {
	weights, err := loadMatrixFromFile(policyFile)
	if err != nil {
		return nil, err
	}

	values, err := loadMatrixFromFile(valueFile)
	if err != nil {
		return nil, err
	}

	return &ActorCritic{weights: weights, values: values}, nil
}

// Helper to save a matrix to a CSV file
func saveMatrixToFile(matrix *mat.Dense, fileName string) error {
	rows, cols := matrix.Dims()
	file, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	for i := 0; i < rows; i++ {
		row := make([]string, cols)
		for j := 0; j < cols; j++ {
			row[j] = strconv.FormatFloat(matrix.At(i, j), 'f', 6, 64)
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}
	return nil
}

// Helper to load a matrix from a CSV file
func loadMatrixFromFile(fileName string) (*mat.Dense, error) {
	file, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	numRows := len(rows)
	numCols := len(rows[0])
	data := make([]float64, numRows*numCols)

	for i, row := range rows {
		for j, val := range row {
			data[i*numCols+j], err = strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, err
			}
		}
	}

	return mat.NewDense(numRows, numCols, data), nil
}

// Save the model as a binary file
func SaveModelBinary(model *ActorCritic, filename string) error {
	// Create file
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Serialize the model using gob
	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(model); err != nil {
		return err
	}

	fmt.Println("Model saved as binary file:", filename)
	return nil
}

// Load the model from a binary file
func LoadModelBinary(filename string) (*ActorCritic, error) {
	// Open file
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Deserialize the model using gob
	var model ActorCritic
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&model); err != nil {
		return nil, err
	}

	fmt.Println("Model loaded from binary file:", filename)
	return &model, nil
}

//a2c required functions end
// a2c scheduler
func (sch *scheduler) selectPatha2c(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
        //fmt.Println(time.Since(s.sessionCreationTime))
	//if sch.actor == nil || sch.critic == nil {
	//var model *ActorCritic // Declare model outside the if bloc
	if sch.aa != 0 {
		fmt.Println("CALLING SETUP A2C")
		fmt.Println(sch.SchedulerName)
		model := CreateModel()
		//sch.setup()
		
		fmt.Println("Model created:", model) // Use the model to avoid 'not used' error

	}

	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		//fmt.Println("<=1")
		return s.paths[protocol.InitialPathID]
		
	}
	
	state := []float64{}
	sRTT := make(map[protocol.PathID]time.Duration)

	//Check for available paths
	var availablePaths []protocol.PathID
	for pathID, path := range s.paths {
		if path.sentPacketHandler.SendingAllowed() && pathID != protocol.InitialPathID {
			availablePaths = append(availablePaths, pathID)
			//fmt.Println("Available: ", pathID)
		}
		if pathID != protocol.InitialPathID {
			RTTs := path.rttStats.LatestRTT()
			CWND := path.sentPacketHandler.GetCongestionWindow()
			Inf := path.sentPacketHandler.GetBytesInFlight()
			/*BSend, err := s.flowControlManager.SendWindowSize(protocol.StreamID(5))
			if err != nil{
				fmt.Print("error in getting sendwin size: ")
				//fmt.Print(s.flowControlManager.streamFlowController)
				fmt.Println(err)
			}*/
			state = append(state, float64(RTTs.Microseconds())/1000)
			state = append(state, float64(CWND)/1000)
			state = append(state, float64(Inf))
		}
	}
	
	//fmt.Println("pathids len: ", len(pathids))
	if len(availablePaths) == 0 {
		if s.paths[protocol.InitialPathID].SendingAllowed() || hasRetransmission {
		       //fmt.Println("==0")
			return s.paths[protocol.InitialPathID]
			
		} else {
			return nil
		}
	}

	if len(availablePaths) == 1 {
		// TODO: Find a way to wait/transmit	 when only one path is available
		//fmt.Println("==1")
		return s.paths[availablePaths[0]]
	}
		//policyFile := "a2c_policy_weights.csv"
        	//valueFile := "a2c_value_weights.csv"
      //var states []*mat.VecDense
      var actions []int
      var rewards []float64  

if model == nil {
	// Try to load the model
	model, err := LoadModelBinary("model.bin")
	if err != nil {
		// If loading fails, create a new model
		fmt.Println("Error loading model:", err)
		model = CreateModel()
		fmt.Println("Created new model.")
	} else {
		// Successfully loaded model
		fmt.Println("Loaded existing model.")
		// Print loaded model details
		fmt.Printf("Loaded Model: %+v\n", model)
	}
}


	        //model = CreateModel()
	        //fmt.Println("Model createrd:", model) // Use the model to avoid 'not used' error
	        
	        //fmt.Println("Model createeed:", model) // Use the model to avoid 'not used' error
		input:=mat.NewVecDense(stateDim, state) //[]float64{p1RTT, p1CWND, p1Inflight, p2RTT, p2CWND, p2Inflight})
		// Select action based on the current state
		//selectedAction := SelectAction(model, input)

		// Compute reward for the selected action
		//reward := ComputeReward(input, selectedAction)

		// Print the results
		//fmt.Printf("Input Set %d:\n", i+1)
		//fmt.Printf("  Selected Path: %d\n", selectedAction+1)
		//fmt.Printf("  Reward: %.2f\n\n", reward)
		
		// Select action
		action := SelectAction(model, input)
		actions = append(actions, action)
		// Compute reward
		duration := time.Since(s.sessionCreationTime)
		var maxRTT time.Duration
		for pathID := range sRTT {
			if sRTT[pathID] > maxRTT {
				maxRTT = sRTT[pathID]
			}
		}
		
		reward := float64(RewardFinalGoodput(sch, s, duration, maxRTT))		
		//reward := ComputeReward(input, action)
		rewards = append(rewards, reward)
		discountedRewards := CalculateDiscountedRewards(rewards)

		// Update the model
		//for t := 0; t < len(states); t++ {
			//UpdateModel(model, input, actions[0], rewards[0], discountedRewards[0])
			UpdateModel(model, input, action, discountedRewards[0])
			//fmt.Println(model.weights.RawMatrix().Data)
			//fmt.Println("updating model")
			//fmt.Println("updating model-----")
		//}
	// Save the model after training
	//if err := SaveModel(model, policyFile, valueFile); err != nil {
	//	fmt.Println("Error saving model:", err)
	//} else {
	//	fmt.Println("Model saved successfully.")
	//}
		
	
	if action < len(availablePaths) {
	//if finalAction-1 >= 0 && finalAction-1 < len(availablePaths) {

		selectedPathID := availablePaths[action]
		//fmt.Println("CALLING")
		return s.paths[selectedPathID]
	}
	
	return nil	
}




//kumar functions start
//kumar functions start
func kumar_getAgents() []Agent {
	agents := []Agent{
		Agent{
			Name:      "Agent1", // High throughput (DDPG)
			wRTT:      -0.3,
			wCWND:     1.5,
			wInflight: 1.5,
		},
		Agent{
			Name:      "Agent2", // Low latency (PPO)
			wRTT:      -1.5,
			wCWND:     0.3,
			wInflight: 0.3,
		},
		Agent{
			Name:      "Agent3", // Balanced (SAC)
			wRTT:      -0.5,
			wCWND:     1.0,
			wInflight: 1.0,
		},
	}
	return agents
}

// computeReward calculates the reward for a path based on the agent's weights.
func (agent *Agent) kumar_computeReward(path []float64) float64 {
	reward := agent.wRTT*path[0] + agent.wCWND*path[1] + agent.wInflight*path[2]
	return reward
}

// computePPMScore calculates the PPM score for a path.
func kumar_computePPMScore(path []float64) float64 {
	score := PPM_wRTT*path[0] + PPM_wCWND*path[1] + PPM_wInflight*path[2]
	return score
}

// assignAgentWeights assigns weights to agents based on the PPM's preferred path.
func kumar_assignAgentWeights(preferredPath int, agents []Agent, state []float64) map[string]float64 {
	weights := make(map[string]float64)

	// Compute differences in metrics
	deltaRTT := state[0] - state[3]
	deltaCWND := state[1] - state[4]
	deltaInflight := state[2] - state[5]

	// Compute contributions to PPM score difference
	deltaScoreRTT := PPM_wRTT * deltaRTT
	deltaScoreCWND := PPM_wCWND * deltaCWND
	deltaScoreInflight := PPM_wInflight * deltaInflight

	// Determine which metric contributed most
	absDeltaScoreRTT := math.Abs(deltaScoreRTT)
	absDeltaScoreCWND := math.Abs(deltaScoreCWND)
	absDeltaScoreInflight := math.Abs(deltaScoreInflight)
	totalDeltaScore := absDeltaScoreRTT + absDeltaScoreCWND + absDeltaScoreInflight

	if totalDeltaScore == 0 {
		// No difference, assign equal weights
		for _, agent := range agents {
			weights[agent.Name] = 1.0 / float64(len(agents))
		}
		return weights
	}

	// Compute proportion of each metric
	propRTT := absDeltaScoreRTT / totalDeltaScore
	propCWND := absDeltaScoreCWND / totalDeltaScore
	propInflight := absDeltaScoreInflight / totalDeltaScore

	// Assign weights to agents
	for _, agent := range agents {
		var weight float64
		switch agent.Name {
		case "Agent1":
			// High throughput (CWND and Inflight)
			weight = propCWND + propInflight
		case "Agent2":
			// Low latency (RTT)
			weight = propRTT
		case "Agent3":
			// Balanced
			weight = (propRTT + propCWND + propInflight) / 3.0
		}
		weights[agent.Name] = weight
	}

	// Normalize weights
	totalWeight := 0.0
	for _, w := range weights {
		totalWeight += w
	}
	for k, w := range weights {
		weights[k] = w / totalWeight
	}

	return weights
}

// combineAgentActions combines the agents' actions based on their confidence and weights.
func kumar_combineAgentActions(agentActions map[string]int, agentConfidences map[string]float64, agentWeights map[string]float64) int {
	pathScores := make(map[int]float64)
	for agentName, chosenPath := range agentActions {
		confidence := agentConfidences[agentName]
		weight := agentWeights[agentName]
		score := confidence * weight

		pathScores[chosenPath] += score
	}

	// Choose the path with the highest score
	var finalAction int
	maxScore := -1.0
	for path, score := range pathScores {
		if score > maxScore {
			maxScore = score
			finalAction = path
		}
	}
	return finalAction
}

// computeGlobalReward calculates the reward for the selected path using the global reward function.
func kumar_computeGlobalReward(state []float64, selectedPath int) float64 {
	var pathIndex int
	if selectedPath == 1 {
		pathIndex = 0
	} else {
		pathIndex = 3
	}

	reward := PPM_wRTT*state[pathIndex+0] + PPM_wCWND*state[pathIndex+1] + PPM_wInflight*state[pathIndex+2]
	return reward
}

//kumar functions end

// kumar scheduler
func (sch *scheduler) selectPathkumar(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	//if sch.actor == nil || sch.critic == nil {
	if sch.x != 0 {
		fmt.Println("CALLING SETUP")
		fmt.Println(sch.SchedulerName)
		sch.setup()
	}

	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}

	state := []float64{}

	//Check for available paths
	var availablePaths []protocol.PathID
	for pathID, path := range s.paths {
		if path.sentPacketHandler.SendingAllowed() && pathID != protocol.InitialPathID {
			availablePaths = append(availablePaths, pathID)
		}
		if pathID != protocol.InitialPathID {
			RTTs := path.rttStats.LatestRTT()
			CWND := path.sentPacketHandler.GetCongestionWindow()
			Inf := path.sentPacketHandler.GetBytesInFlight()
			state = append(state, float64(RTTs.Microseconds())/1000)
			state = append(state, float64(CWND)/1000)
			state = append(state, float64(Inf))
		}
	}
        
	if len(availablePaths) == 0 {
		if s.paths[protocol.InitialPathID].SendingAllowed() || hasRetransmission {
			return s.paths[protocol.InitialPathID]
		} else {
			return nil
		}
	}
        
	if len(availablePaths) == 1 {
		return s.paths[availablePaths[0]]
	}
        //fmt.Println("AP", availablePaths)
	//kumar starts
	agents := kumar_getAgents()

	// Variables to keep track of rewards and actions
	cumulativeReward := 0.0
	finalAction := 1

	// PPM scores
	path1PPMScore := kumar_computePPMScore(state)
	path2PPMScore := kumar_computePPMScore(state)
	//fmt.Printf("state: %v\n", state)

	// Determine PPM preferred path
	var preferredPath int
	if path1PPMScore > path2PPMScore {
		preferredPath = 1
	} else {
		preferredPath = 2
	}

	//fmt.Printf("PPM Preferred Path: %d\n", preferredPath)

	// Agents' decisions and confidence
	agentActions := make(map[string]int)      // Agent name to chosen path
	agentConfidences := make(map[string]float64)

	for _, agent := range agents {
		reward1 := agent.kumar_computeReward(state)
		reward2 := agent.kumar_computeReward(state)

		var chosenPath int
		if reward1 > reward2 {
			chosenPath = 1
		} else {
			chosenPath = 2
		}

		confidence := math.Abs(reward1 - reward2)

		agentActions[agent.Name] = chosenPath
		agentConfidences[agent.Name] = confidence

		//fmt.Printf("%s chose Path %d with confidence %.4f\n", agent.Name, chosenPath, confidence)
	}

	// Assign weights to agents based on PPM
	agentWeights := kumar_assignAgentWeights(preferredPath, agents, state)

	// Combine agents' actions
	finalAction = kumar_combineAgentActions(agentActions, agentConfidences, agentWeights)

	//fmt.Printf("Final selected Path: %d\n", finalAction)

	// Compute reward for the selected path (using a global reward function)
	finalReward := kumar_computeGlobalReward(state, finalAction)

	//fmt.Printf("Reward for selected Path: %.4f\n\n", finalReward)

	cumulativeReward += finalReward

	//fmt.Printf("Cumulative Reward: %.4f\n", cumulativeReward)
	//fmt.Println("A Final selected Path: %d\n", finalAction)
	//fmt.Println("AP", availablePaths)
	if finalAction-1 < len(availablePaths) {
		selectedPathID := availablePaths[finalAction-1]
		return s.paths[selectedPathID]
	}
	return nil

//kumar ends

}

//pumar starts
func pumar_getAgents() []Agent {
	agents := []Agent{
		Agent{
			Name:      "Agent1", // High throughput (DDPG)
			wRTT:      -0.3,
			wCWND:     1.5,
			wInflight: 1.5,
		},
		Agent{
			Name:      "Agent2", // Low latency (PPO)
			wRTT:      -1.5,
			wCWND:     0.3,
			wInflight: 0.3,
		},
		Agent{
			Name:      "Agent3", // Balanced (SAC)
			wRTT:      -0.5,
			wCWND:     1.0,
			wInflight: 1.0,
		},
		
		/*Agent{
			Name:      "Agent1", // High throughput (DDPG)
			wRTT:      -0.5,
			wCWND:     1.0,
			wInflight: 1.0,
		},
		Agent{
			Name:      "Agent2", // Low latency (PPO)
			wRTT:      -1.0,
			wCWND:     0.5,
			wInflight: 0.5,
		},
		Agent{
			Name:      "Agent3", // Balanced (SAC)
			wRTT:      -0.7,
			wCWND:     0.8,
			wInflight: 0.8,
		},*/

	}
	return agents
}
func pumar_assignAgentWeights(preferredPath int, agents []Agent, state []float64) map[string]float64 {
	weights := make(map[string]float64)

	// Compute differences in metrics
	deltaRTT := state[0] - state[3]
	deltaCWND := state[1] - state[4]
	deltaInflight := state[2] - state[5]

	// Compute contributions to PPM score difference
	deltaScoreRTT := PPM_wRTT * deltaRTT
	deltaScoreCWND := PPM_wCWND * deltaCWND
	deltaScoreInflight := PPM_wInflight * deltaInflight

	// Determine which metric contributed most
	absDeltaScoreRTT := math.Abs(deltaScoreRTT)
	absDeltaScoreCWND := math.Abs(deltaScoreCWND)
	absDeltaScoreInflight := math.Abs(deltaScoreInflight)
	totalDeltaScore := absDeltaScoreRTT + absDeltaScoreCWND + absDeltaScoreInflight

	if totalDeltaScore == 0 {
		// No difference, assign equal weights
		for _, agent := range agents {
			weights[agent.Name] = 1.0 / float64(len(agents))
		}
		return weights
	}

	// Compute proportion of each metric
	propRTT := absDeltaScoreRTT / totalDeltaScore
	propCWND := absDeltaScoreCWND / totalDeltaScore
	propInflight := absDeltaScoreInflight / totalDeltaScore

	// Assign weights to agents
	for _, agent := range agents {
		var weight float64
		switch agent.Name {
		case "Agent1":
			// High throughput (CWND and Inflight)
			weight = propCWND + propInflight
		case "Agent2":
			// Low latency (RTT)
			weight = propRTT
		case "Agent3":
			// Balanced
			weight = (propRTT + propCWND + propInflight) / 3.0
		}
		weights[agent.Name] = weight
	}

	// Normalize weights
	totalWeight := 0.0
	for _, w := range weights {
		totalWeight += w
	}
	for k, w := range weights {
		weights[k] = w / totalWeight
	}

	return weights
}
// Normalize the state to prevent skewed reward computations
func normalizeState(state []float64) []float64 {
	maxValues := []float64{1000, 1000, 1000} // Assumed maximum RTT, CWND, and Inflight
	normalized := make([]float64, len(state))
	for i, val := range state {
		normalized[i] = val / maxValues[i%3]
	}
	return normalized
}

// Updated computeReward with normalized state
func (agent *Agent) pumar_computeReward(path []float64) float64 {
	normalizedPath := normalizeState(path)
	reward := agent.wRTT*normalizedPath[0] + agent.wCWND*normalizedPath[1] + agent.wInflight*normalizedPath[2]
	return reward
}

// Updated combineAgentActions with softmax scaling
func pumar_combineAgentActions(agentActions map[string]int, agentConfidences map[string]float64, agentWeights map[string]float64) int {
	pathScores := make(map[int]float64)
	totalConfidence := 0.0

	// Compute weighted scores
	for agentName, chosenPath := range agentActions {
		confidence := agentConfidences[agentName]
		weight := agentWeights[agentName]
		totalConfidence += confidence
		pathScores[chosenPath] += confidence * weight
	}

	// Normalize scores using softmax for better distinction
	maxScore := -math.MaxFloat64
	for _, score := range pathScores {
		if score > maxScore {
			maxScore = score
		}
	}
	sumExpScores := 0.0
	for path, score := range pathScores {
		pathScores[path] = math.Exp(score - maxScore)
		sumExpScores += pathScores[path]
	}
	for path := range pathScores {
		pathScores[path] /= sumExpScores
	}

	// Select path with the highest normalized score
	var finalAction int
	maxNormalizedScore := -1.0
	for path, normalizedScore := range pathScores {
		if normalizedScore > maxNormalizedScore {
			maxNormalizedScore = normalizedScore
			finalAction = path
		}
	}
	return finalAction
}

// Update scheduler to use normalized state
func (sch *scheduler) selectPathpumar(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	// Similar setup logic as before
	//if sch.actor == nil || sch.critic == nil {
	if sch.k == 1 {
		fmt.Println("CALLING SETUP")
		fmt.Println(sch.SchedulerName)
		sch.setup()
	}
        
 
	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}

	state := []float64{}

	//Check for available paths
	var availablePaths []protocol.PathID
	for pathID, path := range s.paths {
		if path.sentPacketHandler.SendingAllowed() && pathID != protocol.InitialPathID {
			availablePaths = append(availablePaths, pathID)
		}
		if pathID != protocol.InitialPathID {
			RTTs := path.rttStats.LatestRTT()
			CWND := path.sentPacketHandler.GetCongestionWindow()
			Inf := path.sentPacketHandler.GetBytesInFlight()
			state = append(state, float64(RTTs.Microseconds())/1000)
			state = append(state, float64(CWND)/1000)
			state = append(state, float64(Inf))
		}
	}
        
	if len(availablePaths) == 0 {
		if s.paths[protocol.InitialPathID].SendingAllowed() || hasRetransmission {
			return s.paths[protocol.InitialPathID]
		} else {
			return nil
		}
	}
        
	if len(availablePaths) == 1 {
		return s.paths[availablePaths[0]]
	}	
       fmt.Println("AP", availablePaths) 
	// Normalize state
	state = normalizeState(state)

	agents := pumar_getAgents()
	agentActions := make(map[string]int)
	agentConfidences := make(map[string]float64)

	for _, agent := range agents {
		reward1 := agent.pumar_computeReward(state)
		reward2 := agent.pumar_computeReward(state)
		chosenPath := 1
		if reward2 > reward1 {
			chosenPath = 2
		}
		confidence := math.Abs(reward1 - reward2)
		agentActions[agent.Name] = chosenPath
		agentConfidences[agent.Name] = confidence
	}

	agentWeights := pumar_assignAgentWeights(1, agents, state) // Assume preferredPath = 1 for simplicity
	finalAction := pumar_combineAgentActions(agentActions, agentConfidences, agentWeights)
        fmt.Println("CALLING", finalAction)
       
	// Path selection logic remains unchanged
	//if finalAction-1 < len(state)/3 {
	if finalAction < len(availablePaths) {
	//if finalAction-1 >= 0 && finalAction-1 < len(availablePaths) {

		selectedPathID := availablePaths[finalAction-1]
		fmt.Println("CALLING")
		return s.paths[selectedPathID]
	}
	
	return nil
}





//pumarends

var jkdmac2 float64 = 0
var rewardmac2 float64 = 0
var critmac2 int =0
// actorm2d_criticm2ab scheduler
func (sch *scheduler) selectPathmac2(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
       //fmt.Println("DEAR MAC 2")
        // fmt.Println(time.Since(s.sessionCreationTime))
	if sch.actorm2d == nil || sch.criticm2a == nil || sch.criticm2b == nil {
		fmt.Println("CALLING SETUP")
		fmt.Println(sch.SchedulerName)
		sch.setup()
	}
	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}
	state := []float64{}
	sRTT := make(map[protocol.PathID]time.Duration)
	//Check for available paths
	var availablePaths []protocol.PathID
	for pathID, path := range s.paths {
		if path.sentPacketHandler.SendingAllowed() && pathID != protocol.InitialPathID {
			availablePaths = append(availablePaths, pathID)
			//fmt.Println("Available: ", pathID)
		}
		if pathID != protocol.InitialPathID {
			RTTs := path.rttStats.LatestRTT()
			CWND := path.sentPacketHandler.GetCongestionWindow()
			Inf := path.sentPacketHandler.GetBytesInFlight()
			/*BSend, err := s.flowControlManager.SendWindowSize(protocol.StreamID(5))
			if err != nil{
				fmt.Print("error in getting sendwin size: ")
				//fmt.Print(s.flowControlManager.streamFlowController)
				fmt.Println(err)
			}*/
			state = append(state, float64(RTTs.Microseconds())/1000)
			state = append(state, float64(CWND)/1000)
			state = append(state, float64(Inf))
			//fmt.Print(float64(RTTs.Microseconds())/1000,",",float64(CWND)/1000,",",float64(Inf),",")
		}
	}
	//fmt.Println("pathids len: ", len(pathids))
	if len(availablePaths) == 0 {
		if s.paths[protocol.InitialPathID].SendingAllowed() || hasRetransmission {
			return s.paths[protocol.InitialPathID]
		} else {
			return nil
		}
	}
	if len(availablePaths) == 1 {
		// TODO: Find a way to wait/transmit when only one path is available
		return s.paths[availablePaths[0]]
	}
	pid := []float64{}
	if sch.Training == false {
		//fmt.Print("state: ")
		//fmt.Println(state)
		//fmt.Print("predicted: ")
		pid = sch.actorm2d.Predict(state)
		//fmt.Println(pid)
	} else {
		/*
		if training:
		-> Get CWND, SWND, InFlight, RTT for both paths i.e. the state
		-> get action_probs & state_value from agent
		-> Get the new_state_value using the agent and find the value function
		-> Take action and calculate rewardmac2 = PS/Tack for a given packet
		-> delta = rewardmac2 + sch.gamma * new_state_value - state_value
		-> actor_loss = -log_prob(Actionprobs) * delta // A list of log action probabilities, multiplied by delta. Used as the loss function for the actor network
		-> critic_loss = delta^2
		-> Use the losses and update the actor and critic networks
		*/
		duration := time.Since(s.sessionCreationTime)
		var maxRTT time.Duration
		for pathID := range sRTT {
			if sRTT[pathID] > maxRTT {
				maxRTT = sRTT[pathID]
			}
		}
		if(sch.pvalm2a == 0 && sch.pvalm2b == 0){
			sch.pvalm2a = sch.criticm2a.Predict(state)[0]
			sch.pvalm2b = sch.criticm2b.Predict(state)[0]
		}
		pid = sch.actorm2d.Predict(state)
		va := (sch.criticm2b.Predict(state))[0]
		sch.ukkm2=0
		//fmt.Println("B--", va)
		if critmac2 == 1 {
		va = (sch.criticm2a.Predict(state))[0]
	        //currentTime := time.Now()
		//formattedTime := currentTime.Format("2006-01-02 15:04:05.999")
		//fmt.Println("A,",formattedTime)
		} else if critmac2 == 0 {
		va = (sch.criticm2b.Predict(state))[0]
		//fmt.Println("B--",va)
	        //currentTime := time.Now()
		//formattedTime := currentTime.Format("2006-01-02 15:04:05.999")
		//fmt.Println("B,",formattedTime)
		}
		//va=va+vb
		rk := float64(RewardFinalGoodputonly(sch, s, duration, maxRTT))
		if(rk>jkdmac2) {
		rewardmac2=1
		//fmt.Print("Yes, ",rk)
		jkdmac2=rk
		}else{
		rewardmac2=-1
		//fmt.Print("No, ",rk)
		if critmac2 == 0 {
		critmac2=1
		} else {
		critmac2 = 0
		}
		}
               //jkdmac2:=0
               //rewardmac2:=rk
		delta := float64(rewardmac2) + sch.gamma*va - sch.pvalm2a;
		//fmt.Print(",",rewardmac2)
		sch.pvalm2a = va;
		actorm2d_loss := make([]float64, len(pid))
		for i, x := range(pid){
			actorm2d_loss[i] = -delta*math.Log(x)
		}
		//critic_loss := delta * delta
		//fmt.Print("calculated losses: ")
		//fmt.Println(actor_loss, critic_loss)
		actorm2d_data := training.Examples{
			training.Example{state, []float64{delta, delta}},
		}
		sch.trainer.Train(sch.actorm2d, actorm2d_data, nil, 1)
		//sch.trainerm2ab.Train(sch.actorm2d, actorm2d_data, nil, 1)
		criticm2a_data := training.Examples{
			training.Example{state, []float64{delta*sch.gamma}},
		}
		sch.trainer.Train(sch.criticm2a, criticm2a_data, nil, 1)
		criticm2b_data := training.Examples{
			training.Example{state, []float64{delta*sch.gamma}},
		}
		sch.trainer.Train(sch.criticm2b, criticm2b_data, nil, 1)
		//sch.trainerm2ab.Train(sch.criticm2a, criticm2a_data, nil, 1)
		/*fmt.Print("state: ")
		fmt.Println(state)
		/fmt.Print("predicted: ")
		fmt.Println(pi)*/
	}

    //fmt.Println("Selecting path %d", selectedPath)

	rand.Seed(time.Now().UTC().UnixNano()) // always seed random!

	pathsToSelect := []wr.Choice{}
	for i, _ := range availablePaths{
		pathsToSelect = append(pathsToSelect, wr.NewChoice(i, uint(pid[i]*100)))
	}
    chooser, _ := wr.NewChooser(
		wr.NewChoice(0, uint(pid[0]*100)),
		wr.NewChoice(1, uint(pid[1]*100)),
	)
	// Choose from prob distribution 
    result := chooser.Pick().(int)
    //fmt.Println("selected choice: ", result)
    //fmt.Print(",",result)
       //fmt.Println(time.Since(s.sessionCreationTime))
	// If both paths are available, decide the path based on the probabilities given by the policy
	return s.paths[availablePaths[result]]
}

var jkm3 float64 = 0
var rewardm3 float64 = 0
var critm3 int =0
// dearmac3
func (sch *scheduler) selectPathmac3(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
       //fmt.Println("DEAR MAC 3")
       //fmt.Println(time.Since(s.sessionCreationTime))

	if sch.actorm3d == nil || sch.criticm3a == nil || sch.criticm3b == nil {
		fmt.Println("CALLING SETUP")
		fmt.Println(sch.SchedulerName)
		sch.setup()
	}
	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}
	state := []float64{}
	sRTT := make(map[protocol.PathID]time.Duration)

	//Check for available paths
	var availablePaths []protocol.PathID
	for pathID, path := range s.paths {
		if path.sentPacketHandler.SendingAllowed() && pathID != protocol.InitialPathID {
			availablePaths = append(availablePaths, pathID)
			//fmt.Println("Available: ", pathID)
		}
		if pathID != protocol.InitialPathID {
			RTTs := path.rttStats.LatestRTT()
			CWND := path.sentPacketHandler.GetCongestionWindow()
			Inf := path.sentPacketHandler.GetBytesInFlight()
			/*BSend, err := s.flowControlManager.SendWindowSize(protocol.StreamID(5))
			if err != nil{
				fmt.Print("error in getting sendwin size: ")
				//fmt.Print(s.flowControlManager.streamFlowController)
				fmt.Println(err)
			}*/
			state = append(state, float64(RTTs.Microseconds())/1000)
			state = append(state, float64(CWND)/1000)
			state = append(state, float64(Inf))
		}
	}
	//fmt.Println("pathids len: ", len(pathids))
	if len(availablePaths) == 0 {
		if s.paths[protocol.InitialPathID].SendingAllowed() || hasRetransmission {
			return s.paths[protocol.InitialPathID]
		} else {
			return nil
		}
	}
	if len(availablePaths) == 1 {
		// TODO: Find a way to wait/transmit when only one path is available
		return s.paths[availablePaths[0]]
	}
	pid := []float64{}
	if sch.Training == false {
		//fmt.Print("state: ")
		//fmt.Println(state)
		//fmt.Print("predicted: ")
		pid = sch.actorm3d.Predict(state)
		//fmt.Println(pid)
	} else {
		/*
		if training:
		-> Get CWND, SWND, InFlight, RTT for both paths i.e. the state
		-> get action_probs & state_value from agent
		-> Get the new_state_value using the agent and find the value function
		-> Take action and calculate rewardm3 = PS/Tack for a given packet
		-> delta = rewardm3 + sch.gamma * new_state_value - state_value
		-> actor_loss = -log_prob(Actionprobs) * delta // A list of log action probabilities, multiplied by delta. Used as the loss function for the actor network
		-> critic_loss = delta^2
		-> Use the losses and update the actor and critic networks
		*/
		duration := time.Since(s.sessionCreationTime)
		var maxRTT time.Duration
		for pathID := range sRTT {
			if sRTT[pathID] > maxRTT {
				maxRTT = sRTT[pathID]
			}
		}
		if(sch.pvalm3a == 0 && sch.pvalm3b == 0){
			sch.pvalm3a = sch.criticm3a.Predict(state)[0]
			sch.pvalm3b = sch.criticm3b.Predict(state)[0]
		}
		pid = sch.actorm3d.Predict(state)
		va := (sch.criticm3b.Predict(state))[0]
		sch.ukkm3=0
		//fmt.Println("B--", va)
		if critm3 == 1 {
		va = (sch.criticm3a.Predict(state))[0]
		//fmt.Println("A--",va)
		} else if critm3 == 0 {
		va = (sch.criticm3b.Predict(state))[0]
		//fmt.Println("B--",va)
		} else if critm3 == 2 {
		va = (sch.criticm3c.Predict(state))[0]
		//fmt.Println("C--",va)
		}
		//va=va+vb
		rk := float64(RewardFinalGoodputonly(sch, s, duration, maxRTT))
		if(rk>jkm3) {
		rewardm3=1
		jkm3=rk
		}else{
		rewardm3=-1
		if critm3 == 0 {
		critm3=1
		} else if critm3 == 1{
		critm3 = 2
		} else {
		critm3=0
		}
		}
               //jkm3:=0
               //rewardm3:=rk
		delta := float64(rewardm3) + sch.gamma*va - sch.pvalm3a;
		//fmt.Println("rewardm3 ",rewardm3)
		sch.pvalm3a = va;
		actord_loss := make([]float64, len(pid))
		for i, x := range(pid){
			actord_loss[i] = -delta*math.Log(x)
		}
		//critic_loss := delta * delta
		//fmt.Print("calculated losses: ")
		//fmt.Println(actor_loss, critic_loss)
		actord_data := training.Examples{
			training.Example{state, []float64{delta, delta}},
		}
		sch.trainer.Train(sch.actorm3d, actord_data, nil, 1)
		//sch.trainerm3ab.Train(sch.actorm3d, actord_data, nil, 1)
		critica_data := training.Examples{
			training.Example{state, []float64{delta*sch.gamma}},
		}
		sch.trainer.Train(sch.criticm3a, critica_data, nil, 1)
		//sch.trainerm3ab.Train(sch.criticm3a, critica_data, nil, 1)
		/*fmt.Print("state: ")
		fmt.Println(state)
		fmt.Print("predicted: ")
		fmt.Println(pi)*/
	}

    //fmt.Println("Selecting path %d", selectedPath)

	rand.Seed(time.Now().UTC().UnixNano()) // always seed random!

	pathsToSelect := []wr.Choice{}
	for i, _ := range availablePaths{
		pathsToSelect = append(pathsToSelect, wr.NewChoice(i, uint(pid[i]*100)))
	}
    chooser, _ := wr.NewChooser(
		wr.NewChoice(0, uint(pid[0]*100)),
		wr.NewChoice(1, uint(pid[1]*100)),
	)
	// Choose from prob distribution 
    result := chooser.Pick().(int)
    //fmt.Println("selected choice: ", result)

	// If both paths are available, decide the path based on the probabilities given by the policy
	return s.paths[availablePaths[result]]
}

var jkcap float64 = 0
var rewardcap float64 = 0
var critcap int =0
// dearcap
func (sch *scheduler) selectPathdearcap(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
        //fmt.Println("CAP")
        //fmt.Println(time.Since(s.sessionCreationTime))

	if sch.actorcapd == nil || sch.criticcapa == nil || sch.criticcapb == nil {
		fmt.Println("CALLING SETUP")
		fmt.Println(sch.SchedulerName)
		sch.setup()
	}
	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}
	state := []float64{}
	sRTT := make(map[protocol.PathID]time.Duration)

	//Check for available paths
	var availablePaths []protocol.PathID
	for pathID, path := range s.paths {
		if path.sentPacketHandler.SendingAllowed() && pathID != protocol.InitialPathID {
			availablePaths = append(availablePaths, pathID)
			//fmt.Println("Available: ", pathID)
		}
		if pathID != protocol.InitialPathID {
			RTTs := path.rttStats.LatestRTT()
			CWND := path.sentPacketHandler.GetCongestionWindow()
			Inf := path.sentPacketHandler.GetBytesInFlight()
			/*BSend, err := s.flowControlManager.SendWindowSize(protocol.StreamID(5))
			if err != nil{
				fmt.Print("error in getting sendwin size: ")
				//fmt.Print(s.flowControlManager.streamFlowController)
				fmt.Println(err)
			}*/
			state = append(state, float64(RTTs.Microseconds())/1000)
			state = append(state, float64(CWND)/1000)
			state = append(state, float64(Inf))
		}
	}
	// Hardcode the split indices
       splitIndex1 := 3
       splitIndex2 := 6

       // Extract values for state1 and state2
       state1 := state[:splitIndex1]
       state2 := state[splitIndex1:splitIndex2]
	//fmt.Println("pathids len: ", len(pathids))
	if len(availablePaths) == 0 {
		if s.paths[protocol.InitialPathID].SendingAllowed() || hasRetransmission {
			return s.paths[protocol.InitialPathID]
		} else {
			return nil
		}
	}
	if len(availablePaths) == 1 {
		// TODO: Find a way to wait/transmit when only one path is available
		return s.paths[availablePaths[0]]
	}
	pid := []float64{}
	if sch.Training == false {
		//fmt.Print("state: ")
		//fmt.Println(state)
		//fmt.Print("predicted: ")
		pid = sch.actorcapd.Predict(state)
		//fmt.Println(pid)
	} else {
		/*
		if training:
		-> Get CWND, SWND, InFlight, RTT for both paths i.e. the state
		-> get action_probs & state_value from agent
		-> Get the new_state_value using the agent and find the value function
		-> Take action and calculate rewardcap = PS/Tack for a given packet
		-> delta = rewardcap + sch.gamma * new_state_value - state_value
		-> actor_loss = -log_prob(Actionprobs) * delta // A list of log action probabilities, multiplied by delta. Used as the loss function for the actor network
		-> critic_loss = delta^2
		-> Use the losses and update the actor and critic networks
		*/
		duration := time.Since(s.sessionCreationTime)
		var maxRTT time.Duration
		for pathID := range sRTT {
			if sRTT[pathID] > maxRTT {
				maxRTT = sRTT[pathID]
			}
		}
		if(sch.pvalcapa == 0 && sch.pvalcapb == 0){
			sch.pvalcapa = sch.criticcapa.Predict(state)[0]
			sch.pvalcapb = sch.criticcapb.Predict(state)[0]
		}
		pid = sch.actorcapd.Predict(state)
		va := (sch.criticcapa.Predict(state1))[0]
		vb := (sch.criticcapb.Predict(state2))[0]
		sch.ukkcap=0
		//fmt.Println("B--", va)
		//if critcap == 1 {
		va = (sch.criticcapa.Predict(state))[0]
		//fmt.Println("A--",va)
		//} else if critcap == 0 {
		//va = (sch.criticcapb.Predict(state))[0]
		//fmt.Println("B--",va)
		
		//}
		va=(va+vb)/2
		rk := float64(RewardFinalGoodputonly(sch, s, duration, maxRTT))
		if(rk>jkcap) {
		rewardcap=1
		jkcap=rk
		}else{
		rewardcap=-1
		if critcap == 0 {
		critcap=1
		} else {
		critcap = 0
		}
		}
               //jkcap:=0
               //rewardcap:=rk
		delta := float64(rewardcap) + sch.gamma*va - sch.pvalcapa;
		//fmt.Println("rewardcap ",rewardcap)
		sch.pvalcapa = va;
		actord_loss := make([]float64, len(pid))
		for i, x := range(pid){
			actord_loss[i] = -delta*math.Log(x)
		}
		//critic_loss := delta * delta
		//fmt.Print("calculated losses: ")
		//fmt.Println(actor_loss, critic_loss)
		actord_data := training.Examples{
			training.Example{state, []float64{delta, delta}},
		}
		sch.trainer.Train(sch.actorcapd, actord_data, nil, 1)
		//sch.trainercapab.Train(sch.actorcapd, actord_data, nil, 1)
		critica_data := training.Examples{
			training.Example{state, []float64{delta*sch.gamma}},
		}
		sch.trainer.Train(sch.criticcapa, critica_data, nil, 1)
		//sch.trainercapab.Train(sch.criticcapa, critica_data, nil, 1)
		/*fmt.Print("state: ")
		fmt.Println(state)
		fmt.Print("predicted: ")
		fmt.Println(pi)*/
	}

    //fmt.Println("Selecting path %d", selectedPath)

	rand.Seed(time.Now().UTC().UnixNano()) // always seed random!

	pathsToSelect := []wr.Choice{}
	for i, _ := range availablePaths{
		pathsToSelect = append(pathsToSelect, wr.NewChoice(i, uint(pid[i]*100)))
	}
    chooser, _ := wr.NewChooser(
		wr.NewChoice(0, uint(pid[0]*100)),
		wr.NewChoice(1, uint(pid[1]*100)),
	)
	// Choose from prob distribution 
    result := chooser.Pick().(int)
    //fmt.Println("selected choice: ", result)

	// If both paths are available, decide the path based on the probabilities given by the policy
	return s.paths[availablePaths[result]]
}



var jkp float64 = 0
var rewardp float64 = 0
var critp int =0
// PACE
func (sch *scheduler) selectPathPACE(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
       // fmt.Println("1")
        // fmt.Println(time.Since(s.sessionCreationTime))
	if sch.actorpa == nil || sch.criticpa == nil || sch.criticpb == nil {
		fmt.Println("CALLING SETUP")
		fmt.Println(sch.SchedulerName)
		sch.setup()
	}
	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}
	state := []float64{}
	sRTT := make(map[protocol.PathID]time.Duration)
        //fmt.Println()
	//Check for available paths
	var availablePaths []protocol.PathID
	for pathID, path := range s.paths {
		if path.sentPacketHandler.SendingAllowed() && pathID != protocol.InitialPathID {
			availablePaths = append(availablePaths, pathID)
			//fmt.Println("Available: ", pathID)
		}
		if pathID != protocol.InitialPathID {
			RTTs := path.rttStats.LatestRTT()
			CWND := path.sentPacketHandler.GetCongestionWindow()
			Inf := path.sentPacketHandler.GetBytesInFlight()
			/*BSend, err := s.flowControlManager.SendWindowSize(protocol.StreamID(5))
			if err != nil{
				fmt.Print("error in getting sendwin size: ")
				//fmt.Print(s.flowControlManager.streamFlowController)
				fmt.Println(err)
			}*/
			state = append(state, float64(RTTs.Microseconds())/1000)
			state = append(state, float64(CWND)/1000)
			state = append(state, float64(Inf))
			//fmt.Print(float64(RTTs.Microseconds())/1000,",",float64(CWND)/1000,",",float64(Inf),",")
		}
	}
	//fmt.Println("pathids len: ", len(pathids))
	if len(availablePaths) == 0 {
		if s.paths[protocol.InitialPathID].SendingAllowed() || hasRetransmission {
			return s.paths[protocol.InitialPathID]
		} else {
			return nil
		}
	}
	if len(availablePaths) == 1 {
		// TODO: Find a way to wait/transmit when only one path is available
		return s.paths[availablePaths[0]]
	}
	pid := []float64{}
	pida := []float64{}
	pidb := []float64{}
	if sch.Training == false {
		//fmt.Print("state: ")
		//fmt.Println(state)
		//fmt.Print("predicted: ")
		pid = sch.actorm2d.Predict(state)
		//fmt.Println(pid)
	} else {
		/*
		if training:
		-> Get CWND, SWND, InFlight, RTT for both paths i.e. the state
		-> get action_probs & state_value from agent
		-> Get the new_state_value using the agent and find the value function
		-> Take action and calculate rewardp = PS/Tack for a given packet
		-> delta = rewardp + sch.gamma * new_state_value - state_value
		-> actor_loss = -log_prob(Actionprobs) * delta // A list of log action probabilities, multiplied by delta. Used as the loss function for the actor network
		-> critic_loss = delta^2
		-> Use the losses and update the actor and critic networks
		*/
		duration := time.Since(s.sessionCreationTime)
		var maxRTT time.Duration
		for pathID := range sRTT {
			if sRTT[pathID] > maxRTT {
				maxRTT = sRTT[pathID]
			}
		}
		if(sch.pvalpa == 0 && sch.pvalpb == 0){
			sch.pvalpa = sch.criticpa.Predict(state)[0]
			sch.pvalpb = sch.criticpb.Predict(state)[0]
		}
		pida = sch.actorpa.Predict(state)
		pidb = sch.actorpb.Predict(state)
                    for i := 0; i < len(pida); i++ {
                           average := (pida[i] + pidb[i]) / 2.0
                           pid = append(pid, average)
                       }

		//fmt.Println("PID", pid)
		//fmt.Println("PIDa", pida)
		//fmt.Println("PIDb", pidb)
		va := (sch.criticpb.Predict(state))[0]
		sch.ukkm2p=0
		//fmt.Println("B--", va)
		if critp == 1 {
		va = (sch.criticpa.Predict(state))[0]
	        //currentTime := time.Now()
		//formattedTime := currentTime.Format("2006-01-02 15:04:05.999")
		//fmt.Println("A,",formattedTime)
		} else if critp == 0 {
		va = (sch.criticpb.Predict(state))[0]
		//fmt.Println("B--",va)
	        //currentTime := time.Now()
		//formattedTime := currentTime.Format("2006-01-02 15:04:05.999")
		//fmt.Println("B,",formattedTime)
		}
		//va=va+vb
		rk := float64(RewardFinalGoodputonly(sch, s, duration, maxRTT))
		if(rk>jkp) {
		rewardp=1
		//fmt.Print("Yes, ",rk)
		jkp=rk
		}else{
		rewardp=-1
		//fmt.Print("No, ",rk)
		if critp == 0 {
		critp=1
		} else {
		critp = 0
		}
		}
               //jkp:=0
               //rewardp:=rk
		delta := float64(rewardp) + sch.gammapab*va - sch.pvalpa;
		//fmt.Print(",",rewardp)
		sch.pvalpa = va;
		actorpa_loss := make([]float64, len(pid))
		for i, x := range(pid){
			actorpa_loss[i] = -delta*math.Log(x)
		}
		//critic_loss := delta * delta
		//fmt.Print("calculated losses: ")
		//fmt.Println(actor_loss, critic_loss)
		actorpa_data := training.Examples{
			training.Example{state, []float64{delta, delta}},
		}
		sch.trainer.Train(sch.actorpa, actorpa_data, nil, 1)
		//sch.trainerpab.Train(sch.actorpa, actorpa_data, nil, 1)
		criticpa_data := training.Examples{
			training.Example{state, []float64{delta*sch.gammapab}},
		}
		sch.trainer.Train(sch.criticpa, criticpa_data, nil, 1)
		criticpb_data := training.Examples{
			training.Example{state, []float64{delta*sch.gammapab}},
		}
		sch.trainer.Train(sch.criticpb, criticpb_data, nil, 1)
		//sch.trainerm2ab.Train(sch.criticm2a, criticm2a_data, nil, 1)
		/*fmt.Print("state: ")
		fmt.Println(state)
		fmt.Print("predicted: ")
		fmt.Println(pi)*/
	}

    //fmt.Println("Selecting path %d", selectedPath)

	rand.Seed(time.Now().UTC().UnixNano()) // always seed random!

	pathsToSelect := []wr.Choice{}
	for i, _ := range availablePaths{
		pathsToSelect = append(pathsToSelect, wr.NewChoice(i, uint(pid[i]*100)))
	}
    chooser, _ := wr.NewChooser(
		wr.NewChoice(0, uint(pid[0]*100)),
		wr.NewChoice(1, uint(pid[1]*100)),
	)
	// Choose from prob distribution 
    result := chooser.Pick().(int)
    //fmt.Println("selected choice: ", result)
    //fmt.Print(",",result)
       //fmt.Println(time.Since(s.sessionCreationTime))
	// If both paths are available, decide the path based on the probabilities given by the policy
	return s.paths[availablePaths[result]]
}




//shravan_scheduler
func (sch *scheduler) selectPathSchShravan(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	if sch.quotas == nil {
		sch.setup()
	}

	// Log Path Id w/ Interface Name
	//for pathId, pth := range s.paths {
	//	fmt.Printf("Path Id: %d, Local Addr: %s, Remote Addr: %s \t", pathId, pth.conn.LocalAddr(), pth.conn.RemoteAddr())
	//}

	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}

	if sch.pathQueue.Empty() {
		for pathId, pth := range s.paths {
			err := sch.pathQueue.Put(queuePathIdItem{pathId: pathId, path: pth})
			if err != nil {
				fmt.Println("Err Inserting in Queue, Error: ", err.Error())
			}
		}
	} else if int64(len(s.paths)) != sch.pathQueue.Len() {
		sch.pathQueue.Get(sch.pathQueue.Len())
		for pathId, pth := range s.paths {
			err := sch.pathQueue.Put(queuePathIdItem{pathId: pathId, path: pth})
			if err != nil {
				fmt.Println("Err Inserting in Queue, Error: ", err.Error())
			}
		}
	}

pathLoop:
	for pathID, pth := range s.paths {
		pathIdFromQueue, _ := sch.pathQueue.Peek()
		pathIdObj, ok := pathIdFromQueue.(queuePathIdItem)
		if !ok {
			panic("Invalid Interface Type Chosen")
		}

		// Don't block path usage if we retransmit, even on another path
		// If this path is potentially failed, do no consider it for sending
		// XXX Prevent using initial pathID if multiple paths
		if (!hasRetransmission && !pth.SendingAllowed()) || pth.potentiallyFailed.Get() || pathID == protocol.InitialPathID {
			if pathIdObj.pathId == pathID {
				_, _ = sch.pathQueue.Get(1)
				_ = sch.pathQueue.Put(pathIdObj)
			}
			continue pathLoop
		}

		if pathIdObj.pathId == pathID {
			_, _ = sch.pathQueue.Get(1)
			_ = sch.pathQueue.Put(pathIdObj)
			return pth
		}
	}
	return nil

}

//shravan_scheduler

func (sch *scheduler) selectPathNeuralNet(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	if sch.quotas == nil {
		sch.setup()
	}

	////Log Path Id w/ Interface Name
	//for pathId, pth := range s.paths {
	//	fmt.Printf("Path Id: %d, Local Addr: %s, Remote Addr: %s \t", pathId, pth.conn.LocalAddr(), pth.conn.RemoteAddr())
	//}

	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	} else if len(s.paths) > 3 {
		return sch.selectPathRoundRobin(s, hasRetransmission, hasStreamRetransmission, fromPth)
	}

	// Load Model

	if sch.pathQueue.Empty() {
		for pathId, pth := range s.paths {
			err := sch.pathQueue.Put(queuePathIdItem{pathId: pathId, path: pth})
			if err != nil {
				fmt.Println("Err Inserting in Queue, Error: ", err.Error())
			}
		}
	} else if int64(len(s.paths)) != sch.pathQueue.Len() {
		sch.pathQueue.Get(sch.pathQueue.Len())
		for pathId, pth := range s.paths {
			err := sch.pathQueue.Put(queuePathIdItem{pathId: pathId, path: pth})
			if err != nil {
				fmt.Println("Err Inserting in Queue, Error: ", err.Error())
			}
		}
	}

pathLoop:
	for pathID, pth := range s.paths {
		pathIdFromQueue, _ := sch.pathQueue.Peek()
		pathIdObj, ok := pathIdFromQueue.(queuePathIdItem)
		if !ok {
			panic("Invalid Interface Type Chosen")
		}

		// Don't block path usage if we retransmit, even on another path
		// If this path is potentially failed, do no consider it for sending
		// XXX Prevent using initial pathID if multiple paths
		if (!hasRetransmission && !pth.SendingAllowed()) || pth.potentiallyFailed.Get() || pathID == protocol.InitialPathID {
			if pathIdObj.pathId == pathID {
				_, _ = sch.pathQueue.Get(1)
				_ = sch.pathQueue.Put(pathIdObj)
			}
			continue pathLoop
		}

		if pathIdObj.pathId == pathID {
			_, _ = sch.pathQueue.Get(1)
			_ = sch.pathQueue.Put(pathIdObj)
			return pth
		}
	}
	return nil
}

func (sch *scheduler) selectPathLowLatency(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	// XXX Avoid using PathID 0 if there is more than 1 path
	fmt.Println(time.Since(s.sessionCreationTime))
	utils.Debugf("selectPathLowLatency")
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			utils.Debugf("Only initial path and sending not allowed without retransmission")
			utils.Debugf("SCH RTT - NIL")
			return nil
		}
		utils.Debugf("Only initial path and sending is allowed or has retransmission")
		utils.Debugf("SCH RTT - InitialPath")
		return s.paths[protocol.InitialPathID]
	}

	// FIXME Only works at the beginning... Cope with new paths during the connection
	if hasRetransmission && hasStreamRetransmission && fromPth.rttStats.SmoothedRTT() == 0 {
		// Is there any other path with a lower number of packet sent?
		currentQuota := sch.quotas[fromPth.pathID]
		for pathID, pth := range s.paths {
			if pathID == protocol.InitialPathID || pathID == fromPth.pathID {
				continue
			}
			// The congestion window was checked when duplicating the packet
			if sch.quotas[pathID] < currentQuota {
				return pth
			}
		}
	}

	var selectedPath *path
	var lowerRTT time.Duration
	var currentRTT time.Duration
	selectedPathID := protocol.PathID(255)

pathLoop:
	for pathID, pth := range s.paths {
		// Don't block path usage if we retransmit, even on another path
		if !hasRetransmission && !pth.SendingAllowed() {
			utils.Debugf("Discarding %d - no has ret and sending is not allowed ", pathID)
			continue pathLoop
		}

		// If this path is potentially failed, do not consider it for sending
		if pth.potentiallyFailed.Get() {
			utils.Debugf("Discarding %d - potentially failed", pathID)
			continue pathLoop
		}

		// XXX Prevent using initial pathID if multiple paths
		if pathID == protocol.InitialPathID {
			continue pathLoop
		}

		currentRTT = pth.rttStats.SmoothedRTT()

		// Prefer staying single-path if not blocked by current path
		// Don't consider this sample if the smoothed RTT is 0
		if lowerRTT != 0 && currentRTT == 0 {
			utils.Debugf("Discarding %d - currentRTT == 0 and lowerRTT != 0 ", pathID)
			continue pathLoop
		}

		// Case if we have multiple paths unprobed
		if currentRTT == 0 {
			currentQuota, ok := sch.quotas[pathID]
			if !ok {
				sch.quotas[pathID] = 0
				currentQuota = 0
			}
			lowerQuota, _ := sch.quotas[selectedPathID]
			if selectedPath != nil && currentQuota > lowerQuota {
				utils.Debugf("Discarding %d - higher quota ", pathID)
				continue pathLoop
			}
		}

		if currentRTT != 0 && lowerRTT != 0 && selectedPath != nil && currentRTT >= lowerRTT {
			utils.Debugf("Discarding %d - higher SRTT ", pathID)
			continue pathLoop
		}

		// Update
		lowerRTT = currentRTT
		selectedPath = pth
		selectedPathID = pathID
	}
        
	return selectedPath
}

func (sch *scheduler) selectBLEST(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}

	// FIXME Only works at the beginning... Cope with new paths during the connection
	if hasRetransmission && hasStreamRetransmission && fromPth.rttStats.SmoothedRTT() == 0 {
		// Is there any other path with a lower number of packet sent?
		currentQuota := sch.quotas[fromPth.pathID]
		for pathID, pth := range s.paths {
			if pathID == protocol.InitialPathID || pathID == fromPth.pathID {
				continue
			}
			// The congestion window was checked when duplicating the packet
			if sch.quotas[pathID] < currentQuota {
				return pth
			}
		}
	}

	var bestPath *path
	var secondBestPath *path
	var lowerRTT time.Duration
	var currentRTT time.Duration
	var secondLowerRTT time.Duration
	bestPathID := protocol.PathID(255)

pathLoop:
	for pathID, pth := range s.paths {
		// Don't block path usage if we retransmit, even on another path
		if !hasRetransmission && !pth.SendingAllowed() {
			continue pathLoop
		}

		// If this path is potentially failed, do not consider it for sending
		if pth.potentiallyFailed.Get() {
			continue pathLoop
		}

		// XXX Prevent using initial pathID if multiple paths
		if pathID == protocol.InitialPathID {
			continue pathLoop
		}

		currentRTT = pth.rttStats.SmoothedRTT()

		// Prefer staying single-path if not blocked by current path
		// Don't consider this sample if the smoothed RTT is 0
		if lowerRTT != 0 && currentRTT == 0 {
			continue pathLoop
		}

		// Case if we have multiple paths unprobed
		if currentRTT == 0 {
			currentQuota, ok := sch.quotas[pathID]
			if !ok {
				sch.quotas[pathID] = 0
				currentQuota = 0
			}
			lowerQuota, _ := sch.quotas[bestPathID]
			if bestPath != nil && currentQuota > lowerQuota {
				continue pathLoop
			}
		}

		if currentRTT >= lowerRTT {
			if (secondLowerRTT == 0 || currentRTT < secondLowerRTT) && pth.SendingAllowed() {
				// Update second best available path
				secondLowerRTT = currentRTT
				secondBestPath = pth
			}
			if currentRTT != 0 && lowerRTT != 0 && bestPath != nil {
				continue pathLoop
			}
		}

		// Update
		lowerRTT = currentRTT
		bestPath = pth
		bestPathID = pathID
	}

	if bestPath == nil {
		if secondBestPath != nil {
			return secondBestPath
		}
		return nil
	}

	if hasRetransmission || bestPath.SendingAllowed() {
		return bestPath
	}

	if secondBestPath == nil {
		return nil
	}
	cwndBest := uint64(bestPath.sentPacketHandler.GetCongestionWindow())
	FirstCo := uint64(protocol.DefaultTCPMSS) * uint64(secondLowerRTT) * (cwndBest*2*uint64(lowerRTT) + uint64(secondLowerRTT) - uint64(lowerRTT))
	BSend, _ := s.flowControlManager.SendWindowSize(protocol.StreamID(5))
	SecondCo := 2 * 1 * uint64(lowerRTT) * uint64(lowerRTT) * (uint64(BSend) - (uint64(secondBestPath.sentPacketHandler.GetBytesInFlight()) + uint64(protocol.DefaultTCPMSS)))

	if FirstCo > SecondCo {
		return nil
	} else {
		return secondBestPath
	}
}

func (sch *scheduler) selectECF(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}

	// FIXME Only works at the beginning... Cope with new paths during the connection
	if hasRetransmission && hasStreamRetransmission && fromPth.rttStats.SmoothedRTT() == 0 {
		// Is there any other path with a lower number of packet sent?
		currentQuota := sch.quotas[fromPth.pathID]
		for pathID, pth := range s.paths {
			if pathID == protocol.InitialPathID || pathID == fromPth.pathID {
				continue
			}
			// The congestion window was checked when duplicating the packet
			if sch.quotas[pathID] < currentQuota {
				return pth
			}
		}
	}

	var bestPath *path
	var secondBestPath *path
	var lowerRTT time.Duration
	var currentRTT time.Duration
	var secondLowerRTT time.Duration
	bestPathID := protocol.PathID(255)

pathLoop:
	for pathID, pth := range s.paths {
		// Don't block path usage if we retransmit, even on another path
		if !hasRetransmission && !pth.SendingAllowed() {
			continue pathLoop
		}

		// If this path is potentially failed, do not consider it for sending
		if pth.potentiallyFailed.Get() {
			continue pathLoop
		}

		// XXX Prevent using initial pathID if multiple paths
		if pathID == protocol.InitialPathID {
			continue pathLoop
		}

		currentRTT = pth.rttStats.SmoothedRTT()

		// Prefer staying single-path if not blocked by current path
		// Don't consider this sample if the smoothed RTT is 0
		if lowerRTT != 0 && currentRTT == 0 {
			continue pathLoop
		}

		// Case if we have multiple paths unprobed
		if currentRTT == 0 {
			currentQuota, ok := sch.quotas[pathID]
			if !ok {
				sch.quotas[pathID] = 0
				currentQuota = 0
			}
			lowerQuota, _ := sch.quotas[bestPathID]
			if bestPath != nil && currentQuota > lowerQuota {
				continue pathLoop
			}
		}

		if currentRTT >= lowerRTT {
			if (secondLowerRTT == 0 || currentRTT < secondLowerRTT) && pth.SendingAllowed() {
				// Update second best available path
				secondLowerRTT = currentRTT
				secondBestPath = pth
			}
			if currentRTT != 0 && lowerRTT != 0 && bestPath != nil {
				continue pathLoop
			}
		}

		// Update
		lowerRTT = currentRTT
		bestPath = pth
		bestPathID = pathID
	}

	if bestPath == nil {
		if secondBestPath != nil {
			return secondBestPath
		}
		return nil
	}

	if hasRetransmission || bestPath.SendingAllowed() {
		return bestPath
	}

	if secondBestPath == nil {
		return nil
	}

	var queueSize uint64
	getQueueSize := func(s *stream) (bool, error) {
		if s != nil {
			queueSize = queueSize + uint64(s.lenOfDataForWriting())
		}
		return true, nil
	}
	s.streamsMap.Iterate(getQueueSize)

	cwndBest := uint64(bestPath.sentPacketHandler.GetCongestionWindow())
	cwndSecond := uint64(secondBestPath.sentPacketHandler.GetCongestionWindow())
	deviationBest := uint64(bestPath.rttStats.MeanDeviation())
	deviationSecond := uint64(secondBestPath.rttStats.MeanDeviation())

	delta := deviationBest
	if deviationBest < deviationSecond {
		delta = deviationSecond
	}
	xBest := queueSize
	if queueSize < cwndBest {
		xBest = cwndBest
	}

	lhs := uint64(lowerRTT) * (xBest + cwndBest)
	rhs := cwndBest * (uint64(secondLowerRTT) + delta)
	if (lhs * 4) < ((rhs * 4) + sch.waiting*rhs) {
		xSecond := queueSize
		if queueSize < cwndSecond {
			xSecond = cwndSecond
		}
		lhsSecond := uint64(secondLowerRTT) * xSecond
		rhsSecond := cwndSecond * (2*uint64(lowerRTT) + delta)
		if lhsSecond > rhsSecond {
			sch.waiting = 1
			return nil
		}
	} else {
		sch.waiting = 0
	}
        fmt.Print(secondBestPath)
	return secondBestPath
}

func (sch *scheduler) selectPathLowBandit(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}

	// FIXME Only works at the beginning... Cope with new paths during the connection
	if hasRetransmission && hasStreamRetransmission && fromPth.rttStats.SmoothedRTT() == 0 {
		// Is there any other path with a lower number of packet sent?
		currentQuota := sch.quotas[fromPth.pathID]
		for pathID, pth := range s.paths {
			if pathID == protocol.InitialPathID || pathID == fromPth.pathID {
				continue
			}
			// The congestion window was checked when duplicating the packet
			if sch.quotas[pathID] < currentQuota {
				return pth
			}
		}
	}

	var bestPath *path
	var secondBestPath *path
	var lowerRTT time.Duration
	var currentRTT time.Duration
	var secondLowerRTT time.Duration
	bestPathID := protocol.PathID(255)

pathLoop:
	for pathID, pth := range s.paths {
		// If this path is potentially failed, do not consider it for sending
		if pth.potentiallyFailed.Get() {
			continue pathLoop
		}

		// XXX Prevent using initial pathID if multiple paths
		if pathID == protocol.InitialPathID {
			continue pathLoop
		}

		currentRTT = pth.rttStats.SmoothedRTT()

		// Prefer staying single-path if not blocked by current path
		// Don't consider this sample if the smoothed RTT is 0
		if lowerRTT != 0 && currentRTT == 0 {
			continue pathLoop
		}

		// Case if we have multiple paths unprobed
		if currentRTT == 0 {
			currentQuota, ok := sch.quotas[pathID]
			if !ok {
				sch.quotas[pathID] = 0
				currentQuota = 0
			}
			lowerQuota, _ := sch.quotas[bestPathID]
			if bestPath != nil && currentQuota > lowerQuota {
				continue pathLoop
			}
		}

		if currentRTT >= lowerRTT {
			if (secondLowerRTT == 0 || currentRTT < secondLowerRTT) && pth.SendingAllowed() {
				// Update second best available path
				secondLowerRTT = currentRTT
				secondBestPath = pth
			}
			if currentRTT != 0 && lowerRTT != 0 && bestPath != nil {
				continue pathLoop
			}
		}

		// Update
		lowerRTT = currentRTT
		bestPath = pth
		bestPathID = pathID

	}

	//Get reward and Update Aa, ba
	if bestPath != nil && secondBestPath != nil {
		for sch.episoderecord < sch.record {
			// Get reward
			cureNum := uint64(0)
			curereward := float64(0)
			if sch.actionvector[sch.episoderecord] == 0 {
				cureNum = uint64(bestPath.sentPacketHandler.GetLeastUnacked() - 1)
			} else {
				cureNum = uint64(secondBestPath.sentPacketHandler.GetLeastUnacked() - 1)
			}
			if sch.packetvector[sch.episoderecord] <= cureNum {
				curereward = float64(protocol.DefaultTCPMSS) / float64(time.Since(sch.zz[sch.episoderecord]))
			} else {
				break
			}
			//Update Aa, ba
			feature := mat.NewDense(banditDimension, 1, nil)
			feature.Set(0, 0, sch.featureone[sch.episoderecord])
			feature.Set(1, 0, sch.featuretwo[sch.episoderecord])
			feature.Set(2, 0, sch.featurethree[sch.episoderecord])
			feature.Set(3, 0, sch.featurefour[sch.episoderecord])
			feature.Set(4, 0, sch.featurefive[sch.episoderecord])
			feature.Set(5, 0, sch.featuresix[sch.episoderecord])

			if sch.actionvector[sch.episoderecord] == 0 {
				rewardMul := mat.NewDense(banditDimension, 1, nil)
				rewardMul.Scale(curereward, feature)
				baF := mat.NewDense(banditDimension, 1, nil)
				for i := 0; i < banditDimension; i++ {
					baF.Set(i, 0, sch.MbaF[i])
				}
				baF.Add(baF, rewardMul)
				for i := 0; i < banditDimension; i++ {
					sch.MbaF[i] = baF.At(i, 0)
				}
				featureMul := mat.NewDense(banditDimension, banditDimension, nil)
				featureMul.Product(feature, feature.T())
				AaF := mat.NewDense(banditDimension, banditDimension, nil)
				for i := 0; i < banditDimension; i++ {
					for j := 0; j < banditDimension; j++ {
						AaF.Set(i, j, sch.MAaF[i][j])
					}
				}
				AaF.Add(AaF, featureMul)
				for i := 0; i < banditDimension; i++ {
					for j := 0; j < banditDimension; j++ {
						sch.MAaF[i][j] = AaF.At(i, j)
					}
				}
				sch.fe += 1
			} else {
				rewardMul := mat.NewDense(banditDimension, 1, nil)
				rewardMul.Scale(curereward, feature)
				baS := mat.NewDense(banditDimension, 1, nil)
				for i := 0; i < banditDimension; i++ {
					baS.Set(i, 0, sch.MbaS[i])
				}
				baS.Add(baS, rewardMul)
				for i := 0; i < banditDimension; i++ {
					sch.MbaS[i] = baS.At(i, 0)
				}
				featureMul := mat.NewDense(banditDimension, banditDimension, nil)
				featureMul.Product(feature, feature.T())
				AaS := mat.NewDense(banditDimension, banditDimension, nil)
				for i := 0; i < banditDimension; i++ {
					for j := 0; j < banditDimension; j++ {
						AaS.Set(i, j, sch.MAaS[i][j])
					}
				}
				AaS.Add(AaS, featureMul)
				for i := 0; i < banditDimension; i++ {
					for j := 0; j < banditDimension; j++ {
						sch.MAaS[i][j] = AaS.At(i, j)
					}
				}
				sch.se += 1
			}
			//Update pointer
			sch.episoderecord += 1
		}
	}

	if bestPath == nil {
		if secondBestPath != nil {
			return secondBestPath
		}
		if s.paths[protocol.InitialPathID].SendingAllowed() || hasRetransmission {
			return s.paths[protocol.InitialPathID]
		} else {
			return nil
		}
	}
	if bestPath.SendingAllowed() {
		sch.waiting = 0
		return bestPath
	}
	if secondBestPath == nil {
		if s.paths[protocol.InitialPathID].SendingAllowed() || hasRetransmission {
			return s.paths[protocol.InitialPathID]
		} else {
			return nil
		}
	}

	if hasRetransmission && secondBestPath.SendingAllowed() {
		return secondBestPath
	}
	if hasRetransmission {
		return s.paths[protocol.InitialPathID]
	}

	if sch.waiting == 1 {
		return nil
	} else {
		// Migrate from buffer to local variables
		AaF := mat.NewDense(banditDimension, banditDimension, nil)
		for i := 0; i < banditDimension; i++ {
			for j := 0; j < banditDimension; j++ {
				AaF.Set(i, j, sch.MAaF[i][j])
			}
		}
		AaS := mat.NewDense(banditDimension, banditDimension, nil)
		for i := 0; i < banditDimension; i++ {
			for j := 0; j < banditDimension; j++ {
				AaS.Set(i, j, sch.MAaS[i][j])
			}
		}
		baF := mat.NewDense(banditDimension, 1, nil)
		for i := 0; i < banditDimension; i++ {
			baF.Set(i, 0, sch.MbaF[i])
		}
		baS := mat.NewDense(banditDimension, 1, nil)
		for i := 0; i < banditDimension; i++ {
			baS.Set(i, 0, sch.MbaS[i])
		}

		//Features
		cwndBest := float64(bestPath.sentPacketHandler.GetCongestionWindow())
		cwndSecond := float64(secondBestPath.sentPacketHandler.GetCongestionWindow())
		BSend, _ := s.flowControlManager.SendWindowSize(protocol.StreamID(5))
		inflightf := float64(bestPath.sentPacketHandler.GetBytesInFlight())
		inflights := float64(secondBestPath.sentPacketHandler.GetBytesInFlight())
		llowerRTT := bestPath.rttStats.LatestRTT()
		lsecondLowerRTT := secondBestPath.rttStats.LatestRTT()
		feature := mat.NewDense(banditDimension, 1, nil)
		if 0 < float64(lsecondLowerRTT) && 0 < float64(llowerRTT) {
			feature.Set(0, 0, cwndBest/float64(llowerRTT))
			feature.Set(2, 0, float64(BSend)/float64(llowerRTT))
			feature.Set(4, 0, inflightf/float64(llowerRTT))
			feature.Set(1, 0, inflights/float64(lsecondLowerRTT))
			feature.Set(3, 0, float64(BSend)/float64(lsecondLowerRTT))
			feature.Set(5, 0, cwndSecond/float64(lsecondLowerRTT))
		} else {
			feature.Set(0, 0, 0)
			feature.Set(2, 0, 0)
			feature.Set(4, 0, 0)
			feature.Set(1, 0, 0)
			feature.Set(3, 0, 0)
			feature.Set(5, 0, 0)
		}

		//Buffer feature for latter update
		sch.featureone[sch.record] = feature.At(0, 0)
		sch.featuretwo[sch.record] = feature.At(1, 0)
		sch.featurethree[sch.record] = feature.At(2, 0)
		sch.featurefour[sch.record] = feature.At(3, 0)
		sch.featurefive[sch.record] = feature.At(4, 0)
		sch.featuresix[sch.record] = feature.At(5, 0)

		//Obtain theta
		AaIF := mat.NewDense(banditDimension, banditDimension, nil)
		AaIF.Inverse(AaF)
		thetaF := mat.NewDense(banditDimension, 1, nil)
		thetaF.Product(AaIF, baF)

		AaIS := mat.NewDense(banditDimension, banditDimension, nil)
		AaIS.Inverse(AaS)
		thetaS := mat.NewDense(banditDimension, 1, nil)
		thetaS.Product(AaIS, baS)

		//Obtain bandit value
		thetaFPro := mat.NewDense(1, 1, nil)
		thetaFPro.Product(thetaF.T(), feature)
		featureFProOne := mat.NewDense(1, banditDimension, nil)
		featureFProOne.Product(feature.T(), AaIF)
		featureFProTwo := mat.NewDense(1, 1, nil)
		featureFProTwo.Product(featureFProOne, feature)

		thetaSPro := mat.NewDense(1, 1, nil)
		thetaSPro.Product(thetaS.T(), feature)
		featureSProOne := mat.NewDense(1, banditDimension, nil)
		featureSProOne.Product(feature.T(), AaIS)
		featureSProTwo := mat.NewDense(1, 1, nil)
		featureSProTwo.Product(featureSProOne, feature)

		//Make decision based on bandit value
		if (thetaSPro.At(0, 0) + banditAlpha*math.Sqrt(featureSProTwo.At(0, 0))) < (thetaFPro.At(0, 0) + banditAlpha*math.Sqrt(featureFProTwo.At(0, 0))) {
			sch.waiting = 1
			sch.zz[sch.record] = time.Now()
			sch.actionvector[sch.record] = 0
			sch.packetvector[sch.record] = bestPath.sentPacketHandler.GetLastPackets() + 1
			sch.record += 1
			return nil
		} else {
			sch.waiting = 0
			sch.zz[sch.record] = time.Now()
			sch.actionvector[sch.record] = 1
			sch.packetvector[sch.record] = secondBestPath.sentPacketHandler.GetLastPackets() + 1
			sch.record += 1
			return secondBestPath
		}

	}

}

func (sch *scheduler) selectPathPeek(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	// XXX Avoid using PathID 0 if there is more than 1 path
	fmt.Println(time.Since(s.sessionCreationTime))
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}

	// FIXME Only works at the beginning... Cope with new paths during the connection
	if hasRetransmission && hasStreamRetransmission && fromPth.rttStats.SmoothedRTT() == 0 {
		// Is there any other path with a lower number of packet sent?
		currentQuota := sch.quotas[fromPth.pathID]
		for pathID, pth := range s.paths {
			if pathID == protocol.InitialPathID || pathID == fromPth.pathID {
				continue
			}
			// The congestion window was checked when duplicating the packet
			if sch.quotas[pathID] < currentQuota {
				return pth
			}
		}
	}

	var bestPath *path
	var secondBestPath *path
	var lowerRTT time.Duration
	var currentRTT time.Duration
	var secondLowerRTT time.Duration
	bestPathID := protocol.PathID(255)

pathLoop:
	for pathID, pth := range s.paths {
		// If this path is potentially failed, do not consider it for sending
		if pth.potentiallyFailed.Get() {
			continue pathLoop
		}

		// XXX Prevent using initial pathID if multiple paths
		if pathID == protocol.InitialPathID {
			continue pathLoop
		}

		currentRTT = pth.rttStats.SmoothedRTT()

		// Prefer staying single-path if not blocked by current path
		// Don't consider this sample if the smoothed RTT is 0
		if lowerRTT != 0 && currentRTT == 0 {
			continue pathLoop
		}

		// Case if we have multiple paths unprobed
		if currentRTT == 0 {
			currentQuota, ok := sch.quotas[pathID]
			if !ok {
				sch.quotas[pathID] = 0
				currentQuota = 0
			}
			lowerQuota, _ := sch.quotas[bestPathID]
			if bestPath != nil && currentQuota > lowerQuota {
				continue pathLoop
			}
		}

		if currentRTT >= lowerRTT {
			if (secondLowerRTT == 0 || currentRTT < secondLowerRTT) && pth.SendingAllowed() {
				// Update second best available path
				secondLowerRTT = currentRTT
				secondBestPath = pth
			}
			if currentRTT != 0 && lowerRTT != 0 && bestPath != nil {
				continue pathLoop
			}
		}

		// Update
		lowerRTT = currentRTT
		bestPath = pth
		bestPathID = pathID

	}

	// TODO: Support More than 2 best Paths
	if bestPath == nil {
		if secondBestPath != nil {
			return secondBestPath
		}
		if s.paths[protocol.InitialPathID].SendingAllowed() || hasRetransmission {
			return s.paths[protocol.InitialPathID]
		} else {
			return nil
		}
	}
	if bestPath.SendingAllowed() {
		sch.waiting = 0
		return bestPath
	}
	if secondBestPath == nil {
		if s.paths[protocol.InitialPathID].SendingAllowed() || hasRetransmission {
			return s.paths[protocol.InitialPathID]
		} else {
			return nil
		}
	}

	if hasRetransmission && secondBestPath.SendingAllowed() {
		return secondBestPath
	}
	if hasRetransmission {
		return s.paths[protocol.InitialPathID]
	}

	if sch.waiting == 1 {
		return nil
	} else {
		// Migrate from buffer to local variables
		// TODO: Combine 4 loops to single loop
		AaF := mat.NewDense(banditDimension, banditDimension, nil)
		for i := 0; i < banditDimension; i++ {
			for j := 0; j < banditDimension; j++ {
				AaF.Set(i, j, sch.MAaF[i][j])
			}
		}
		AaS := mat.NewDense(banditDimension, banditDimension, nil)
		for i := 0; i < banditDimension; i++ {
			for j := 0; j < banditDimension; j++ {
				AaS.Set(i, j, sch.MAaS[i][j])
			}
		}
		baF := mat.NewDense(banditDimension, 1, nil)
		for i := 0; i < banditDimension; i++ {
			baF.Set(i, 0, sch.MbaF[i])
		}
		baS := mat.NewDense(banditDimension, 1, nil)
		for i := 0; i < banditDimension; i++ {
			baS.Set(i, 0, sch.MbaS[i])
		}

		//Features
		cwndBest := float64(bestPath.sentPacketHandler.GetCongestionWindow())
		cwndSecond := float64(secondBestPath.sentPacketHandler.GetCongestionWindow())
		BSend, _ := s.flowControlManager.SendWindowSize(protocol.StreamID(5))
		inflightf := float64(bestPath.sentPacketHandler.GetBytesInFlight())
		inflights := float64(secondBestPath.sentPacketHandler.GetBytesInFlight())
		llowerRTT := bestPath.rttStats.LatestRTT()
		lsecondLowerRTT := secondBestPath.rttStats.LatestRTT()
		feature := mat.NewDense(banditDimension, 1, nil)
		if 0 < float64(lsecondLowerRTT) && 0 < float64(llowerRTT) {
			feature.Set(0, 0, cwndBest/float64(llowerRTT))
			feature.Set(2, 0, float64(BSend)/float64(llowerRTT))
			feature.Set(4, 0, inflightf/float64(llowerRTT))
			feature.Set(1, 0, inflights/float64(lsecondLowerRTT))
			feature.Set(3, 0, float64(BSend)/float64(lsecondLowerRTT))
			feature.Set(5, 0, cwndSecond/float64(lsecondLowerRTT))
		} else {
			feature.Set(0, 0, 0)
			feature.Set(2, 0, 0)
			feature.Set(4, 0, 0)
			feature.Set(1, 0, 0)
			feature.Set(3, 0, 0)
			feature.Set(5, 0, 0)
		}

		//Obtain theta
		AaIF := mat.NewDense(banditDimension, banditDimension, nil)
		AaIF.Inverse(AaF)
		thetaF := mat.NewDense(banditDimension, 1, nil)
		thetaF.Product(AaIF, baF)

		AaIS := mat.NewDense(banditDimension, banditDimension, nil)
		AaIS.Inverse(AaS)
		thetaS := mat.NewDense(banditDimension, 1, nil)
		thetaS.Product(AaIS, baS)

		//Obtain bandit value
		thetaFPro := mat.NewDense(1, 1, nil)
		thetaFPro.Product(thetaF.T(), feature)

		thetaSPro := mat.NewDense(1, 1, nil)
		thetaSPro.Product(thetaS.T(), feature)

		//Make decision based on bandit value and stochastic value
		if thetaSPro.At(0, 0) < thetaFPro.At(0, 0) {
			if rand.Intn(100) < 70 {
				sch.waiting = 1
				return nil
			} else {
				sch.waiting = 0
				//fmt.Println(time.Since(s.sessionCreationTime))
				return secondBestPath
			}
		} else {
			if rand.Intn(100) < 90 {
				sch.waiting = 0
				//fmt.Println(time.Since(s.sessionCreationTime))
				return secondBestPath
			} else {
				sch.waiting = 1
				return nil
			}
		}
	}

}

func (sch *scheduler) selectPathRandom(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}
	var availablePaths []protocol.PathID

	for pathID, pth := range s.paths {
		cong := float32(pth.sentPacketHandler.GetCongestionWindow()) - float32(pth.sentPacketHandler.GetBytesInFlight())
		allowed := pth.SendingAllowed() || (cong <= 0 && float32(cong) >= -float32(pth.sentPacketHandler.GetCongestionWindow())*float32(sch.AllowedCongestion)*0.01)

		if pathID != protocol.InitialPathID && (allowed || hasRetransmission) {
			//if pathID != protocol.InitialPathID && (pth.SendingAllowed() || hasRetransmission){
			availablePaths = append(availablePaths, pathID)
		}
	}

	if len(availablePaths) == 0 {
		return nil
	}

	pathID := rand.Intn(len(availablePaths))
	utils.Debugf("Selecting path %d", pathID)
	return s.paths[availablePaths[pathID]]
}

func (sch *scheduler) selectFirstPath(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}
	for pathID, pth := range s.paths {
		if pathID == protocol.PathID(1) && pth.SendingAllowed() {
			return pth
		}
	}

	return nil
}

func (sch *scheduler) selectPatDQNAgent(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	// XXX Avoid using PathID 0 if there is more than 1 path
	fmt.Println("1")
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}

	if len(s.paths) == 2 {
		for pathID, path := range s.paths {
			if pathID != protocol.InitialPathID {
				utils.Debugf("Selecting path %d as unique path", pathID)
				return path
			}
		}
	}

	//Check for available paths
	var availablePaths []protocol.PathID
	for pathID, path := range s.paths {
		if path.sentPacketHandler.SendingAllowed() && pathID != protocol.InitialPathID {
			availablePaths = append(availablePaths, pathID)
		}
	}

	if len(availablePaths) == 0 {
		if s.paths[protocol.InitialPathID].SendingAllowed() || hasRetransmission {
			return s.paths[protocol.InitialPathID]
		} else {
			return nil
		}
	} else if len(availablePaths) == 1 {
		return s.paths[availablePaths[0]]
	}

	action, paths := GetStateAndReward(sch, s)

	if paths == nil {
		return s.paths[protocol.InitialPathID]
	}

	return paths[action]
}

// Lock of s.paths must be held
func (sch *scheduler) selectPath(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	switch s.config.Scheduler {
	case constants.SCHEDULER_ROUND_ROBIN:
		return sch.selectPathRoundRobin(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_SCH_SHRAVAN:
		return sch.selectPathSchShravan(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_LOW_LATENCY:
		return sch.selectPathLowLatency(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_FIRST_PATH:
		return sch.selectFirstPath(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_BLEST:
		return sch.selectBLEST(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_AGENT:
		return sch.selectPatDQNAgent(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_ECF:
		return sch.selectECF(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_LOW_BANDIT:
		return sch.selectPathLowBandit(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_PEEKABOO:
		return sch.selectPathPeek(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_RANDOM:
		return sch.selectPathRandom(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_NEURAL_NET:
		return sch.selectPathNeuralNet(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_DEAR:
		return sch.selectPathdear(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_KUMAR:
		return sch.selectPathkumar(s, hasRetransmission, hasStreamRetransmission, fromPth)	
	case constants.SCHEDULER_PUMAR:
		return sch.selectPathpumar(s, hasRetransmission, hasStreamRetransmission, fromPth)	

	case constants.SCHEDULER_MAC2:
		return sch.selectPathmac2(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_MAC3:
		return sch.selectPathmac3(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_CAP:
		return sch.selectPathdearcap(s, hasRetransmission, hasStreamRetransmission, fromPth)
	case constants.SCHEDULER_A2C:
		return sch.selectPatha2c(s, hasRetransmission, hasStreamRetransmission, fromPth)


	case constants.SCHEDULER_PACE:
		return sch.selectPathPACE(s, hasRetransmission, hasStreamRetransmission, fromPth)

	default:
		return sch.selectPathRoundRobin(s, hasRetransmission, hasStreamRetransmission, fromPth)
	}
}

// Lock of s.paths must be free (in case of log print)
func (sch *scheduler) performPacketSending(s *session, windowUpdateFrames []*wire.WindowUpdateFrame, pth *path) (*ackhandler.Packet, bool, error) {
	// add a retransmittable frame
	if pth.sentPacketHandler.ShouldSendRetransmittablePacket() {
		s.packer.QueueControlFrame(&wire.PingFrame{}, pth)
	}
	packet, err := s.packer.PackPacket(pth)
	if err != nil || packet == nil {
		return nil, false, err
	}
	if err = s.sendPackedPacket(packet, pth); err != nil {
		return nil, false, err
	}

	// send every window update twice
	for _, f := range windowUpdateFrames {
		s.packer.QueueControlFrame(f, pth)
	}

	// Packet sent, so update its quota
	sch.quotas[pth.pathID]++

	sRTT := make(map[protocol.PathID]time.Duration)

	// Provide some logging if it is the last packet
	for _, frame := range packet.frames {
		switch frame := frame.(type) {
		case *wire.StreamFrame:
			if frame.FinBit {
				// Last packet to send on the stream, print stats
				s.pathsLock.RLock()
				utils.Infof("Info for stream %x of %x", frame.StreamID, s.connectionID)
				for pathID, pth := range s.paths {
					sntPkts, sntRetrans, sntLost := pth.sentPacketHandler.GetStatistics()
					rcvPkts := pth.receivedPacketHandler.GetStatistics()
					utils.Infof("Path %x: sent %d retrans %d lost %d; rcv %d rtt %v", pathID, sntPkts, sntRetrans, sntLost, rcvPkts, pth.rttStats.SmoothedRTT())
					//utils.Infof("Congestion Window: %d", pth.sentPacketHandler.GetCongestionWindow())
					if sch.Training {
						sRTT[pathID] = pth.rttStats.SmoothedRTT()
					}
				}
				utils.Infof("Action: %d", sch.actionvector)
				utils.Infof("record: %d", sch.record)
				utils.Infof("epsidoe: %d", sch.episoderecord)
				utils.Infof("fe: %d", sch.fe)
				utils.Infof("se: %d", sch.se)
				if sch.Training && sch.SchedulerName == "dqnAgent" {
					duration := time.Since(s.sessionCreationTime)
					var maxRTT time.Duration
					for pathID := range sRTT {
						if sRTT[pathID] > maxRTT {
							maxRTT = sRTT[pathID]
						}
					}
					sch.TrainingAgent.CloseEpisode(uint64(s.connectionID), RewardFinalGoodput(sch, s, duration, maxRTT), false)
				}
				utils.Infof("Dump: %t, Training:%t, scheduler:%s", sch.DumpExp, sch.Training, sch.SchedulerName)
				if sch.DumpExp && !sch.Training && sch.SchedulerName == "dqnAgent" {
					utils.Infof("Closing episode %d", uint64(s.connectionID))
					sch.dumpAgent.CloseExperience(uint64(s.connectionID))
				}
				s.pathsLock.RUnlock()
				//Write lin parameters
				os.Remove(sch.projectHomeDir + "/sch_out/lin")
				os.Create(sch.projectHomeDir + "/sch_out/lin")
				file2, _ := os.OpenFile(sch.projectHomeDir+"/sch_out/lin", os.O_WRONLY, 0600)
				for i := 0; i < banditDimension; i++ {
					for j := 0; j < banditDimension; j++ {
						fmt.Fprintf(file2, "%.8f\n", sch.MAaF[i][j])
					}
				}
				for i := 0; i < banditDimension; i++ {
					for j := 0; j < banditDimension; j++ {
						fmt.Fprintf(file2, "%.8f\n", sch.MAaS[i][j])
					}
				}
				for j := 0; j < banditDimension; j++ {
					fmt.Fprintf(file2, "%.8f\n", sch.MbaF[j])
				}
				for j := 0; j < banditDimension; j++ {
					fmt.Fprintf(file2, "%.8f\n", sch.MbaS[j])
				}
				file2.Close()

				if(sch.Training == true && sch.actor != nil){
					// Log actor(policy) weights
					fmt.Println("Saving weights of actor & critic")
					//start := time.Now()
					actorFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/actorweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					sum := 0
					for i, l := range sch.actor.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(actorFile, "%f\n", sch.actor.Layers[i].Neurons[j].In[k].Weight)
								sum += 1
							}
						}
					}
					actorFile.Close()

					criticFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/criticweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					for i, l := range sch.critic.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(criticFile, "%f\n", sch.critic.Layers[i].Neurons[j].In[k].Weight)
							}
						}
					}
					criticFile.Close()
				}
				//dearmac2
				if(sch.Training == true && sch.actorm2d != nil){
					// Log actor(policy) weights
					fmt.Println("Saving weights of actor & critic")
					//start := time.Now()
					actorm2dFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/actorm2dweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					sum := 0
					for i, l := range sch.actorm2d.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(actorm2dFile, "%f\n", sch.actorm2d.Layers[i].Neurons[j].In[k].Weight)
								sum += 1
							}
						}
					}
					actorm2dFile.Close()

					criticm2aFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/criticm2aweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					for i, l := range sch.criticm2a.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(criticm2aFile, "%f\n", sch.criticm2a.Layers[i].Neurons[j].In[k].Weight)
							}
						}
					}
					criticm2aFile.Close()
					criticm2bFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/criticm2bweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					for i, l := range sch.criticm2b.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(criticm2bFile, "%f\n", sch.criticm2b.Layers[i].Neurons[j].In[k].Weight)
							}
						}
					}
					criticm2bFile.Close()
				}
				
				if(sch.Training == true && sch.SchedulerName == "a2c"){
					// Save the model
					fmt.Println("came in")
	err := SaveModelBinary(model, "model.bin")
	if err != nil {
		fmt.Println("Error saving model:", err)
		//return nil
	}
				
				}				
				
				
				//dearmac3
				if(sch.Training == true && sch.actorm3d != nil){
					// Log actor(policy) weights
					fmt.Println("Saving weights of actor & critic")
					//start := time.Now()
					actorm3dFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/actorm3dweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					sum := 0
					for i, l := range sch.actorm3d.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(actorm3dFile, "%f\n", sch.actorm3d.Layers[i].Neurons[j].In[k].Weight)
								sum += 1
							}
						}
					}
					actorm3dFile.Close()

					criticm3aFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/criticm3aweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					for i, l := range sch.criticm3a.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(criticm3aFile, "%f\n", sch.criticm3a.Layers[i].Neurons[j].In[k].Weight)
							}
						}
					}
					criticm3aFile.Close()
					criticm3bFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/criticm3bweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					for i, l := range sch.criticm3b.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(criticm3bFile, "%f\n", sch.criticm3b.Layers[i].Neurons[j].In[k].Weight)
							}
						}
					}
					criticm3bFile.Close()
					criticm3cFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/criticm3cweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					for i, l := range sch.criticm3c.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(criticm3cFile, "%f\n", sch.criticm3c.Layers[i].Neurons[j].In[k].Weight)
							}
						}
					}
					criticm3cFile.Close()

				}
				//dearcap
				if(sch.Training == true && sch.actorcapd != nil){
					// Log actor(policy) weights
					fmt.Println("Saving weights of actor & critic")
					//start := time.Now()
					actorcapdFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/actorcapdweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					sum := 0
					for i, l := range sch.actorcapd.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(actorcapdFile, "%f\n", sch.actorcapd.Layers[i].Neurons[j].In[k].Weight)
								sum += 1
							}
						}
					}
					actorcapdFile.Close()

					criticcapaFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/criticcapaweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					for i, l := range sch.criticcapa.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(criticcapaFile, "%f\n", sch.criticcapa.Layers[i].Neurons[j].In[k].Weight)
							}
						}
					}
					criticcapaFile.Close()
					criticcapbFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/criticcapbweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					for i, l := range sch.criticcapb.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(criticcapbFile, "%f\n", sch.criticcapb.Layers[i].Neurons[j].In[k].Weight)
							}
						}
					}
					criticcapbFile.Close()
				}
				//PACE
				if(sch.Training == true && sch.actorpa != nil){
					// Log actorpa(policy) weights
					fmt.Println("Saving weights of actor & critic")
					//start := time.Now()
					actorpaFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/actorpaweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					sum := 0
					for i, l := range sch.actorpa.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(actorpaFile, "%f\n", sch.actorpa.Layers[i].Neurons[j].In[k].Weight)
								sum += 1
							}
						}
					}
					actorpaFile.Close()
					// Log actorpb(policy) weights
					fmt.Println("Saving weights of actor & critic")
					//start := time.Now()
					actorpbFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/actorpbweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					sumpb := 0
					for i, l := range sch.actorpb.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(actorpbFile, "%f\n", sch.actorpb.Layers[i].Neurons[j].In[k].Weight)
								sumpb += 1
							}
						}
					}
					actorpbFile.Close()
					criticpaFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/criticpaweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					for i, l := range sch.criticpa.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(criticpaFile, "%f\n", sch.criticpa.Layers[i].Neurons[j].In[k].Weight)
							}
						}
					}
					criticpaFile.Close()
					criticpbFile, err := os.OpenFile(sch.projectHomeDir+"/sch_out/criticpbweights", os.O_WRONLY, 0600)
					if(err != nil){
						fmt.Println("Error in opening file")
						fmt.Println(err)
					}
					for i, l := range sch.criticpb.Layers {
						for j := range l.Neurons {
							for k := range l.Neurons[j].In {
								fmt.Fprintf(criticpbFile, "%f\n", sch.criticpb.Layers[i].Neurons[j].In[k].Weight)
							}
						}
					}
					criticpbFile.Close()
				}
				
			}
		default:
		}
	}

	pkt := &ackhandler.Packet{
		PacketNumber:    packet.number,
		Frames:          packet.frames,
		Length:          protocol.ByteCount(len(packet.raw)),
		EncryptionLevel: packet.encryptionLevel,
	}

	return pkt, true, nil
}

// Lock of s.paths must be free
func (sch *scheduler) ackRemainingPaths(s *session, totalWindowUpdateFrames []*wire.WindowUpdateFrame) error {
	// Either we run out of data, or CWIN of usable paths are full
	// Send ACKs on paths not yet used, if needed. Either we have no data to send and
	// it will be a pure ACK, or we will have data in it, but the CWIN should then
	// not be an issue.
	s.pathsLock.RLock()
	defer s.pathsLock.RUnlock()
	// get WindowUpdate frames
	// this call triggers the flow controller to increase the flow control windows, if necessary
	windowUpdateFrames := totalWindowUpdateFrames
	if len(windowUpdateFrames) == 0 {
		windowUpdateFrames = s.getWindowUpdateFrames(s.peerBlocked)
	}
	for _, pthTmp := range s.paths {
		ackTmp := pthTmp.GetAckFrame()
		for _, wuf := range windowUpdateFrames {
			s.packer.QueueControlFrame(wuf, pthTmp)
		}
		if ackTmp != nil || len(windowUpdateFrames) > 0 {
			if pthTmp.pathID == protocol.InitialPathID && ackTmp == nil {
				continue
			}
			swf := pthTmp.GetStopWaitingFrame(false)
			if swf != nil {
				s.packer.QueueControlFrame(swf, pthTmp)
			}
			s.packer.QueueControlFrame(ackTmp, pthTmp)
			// XXX (QDC) should we instead call PackPacket to provides WUFs?
			var packet *packedPacket
			var err error
			if ackTmp != nil {
				// Avoid internal error bug
				packet, err = s.packer.PackAckPacket(pthTmp)
			} else {
				packet, err = s.packer.PackPacket(pthTmp)
			}
			if err != nil {
				return err
			}
			err = s.sendPackedPacket(packet, pthTmp)
			if err != nil {
				return err
			}
		}
	}
	s.peerBlocked = false
	return nil
}

func (sch *scheduler) sendPacket(s *session) error {
	var pth *path

	// Update leastUnacked value of paths
	s.pathsLock.RLock()
	for _, pthTmp := range s.paths {
		pthTmp.SetLeastUnacked(pthTmp.sentPacketHandler.GetLeastUnacked())
	}
	s.pathsLock.RUnlock()

	// get WindowUpdate frames
	// this call triggers the flow controller to increase the flow control windows, if necessary
	windowUpdateFrames := s.getWindowUpdateFrames(false)
	for _, wuf := range windowUpdateFrames {
		s.packer.QueueControlFrame(wuf, pth)
	}

	// Repeatedly try sending until we don't have any more data, or run out of the congestion window
	for {
		// We first check for retransmissions
		hasRetransmission, retransmitHandshakePacket, fromPth := sch.getRetransmission(s)
		// XXX There might still be some stream frames to be retransmitted
		hasStreamRetransmission := s.streamFramer.HasFramesForRetransmission()

		// Select the path here
		s.pathsLock.RLock()
		pth = sch.selectPath(s, hasRetransmission, hasStreamRetransmission, fromPth)
		go sch.logTrainingData(s, pth, sch.OnlineTrainingFile)
		s.pathsLock.RUnlock()

		// XXX No more path available, should we have a new QUIC error message?
		if pth == nil {
			windowUpdateFrames := s.getWindowUpdateFrames(false)
			return sch.ackRemainingPaths(s, windowUpdateFrames)
		}

		// If we have an handshake packet retransmission, do it directly
		if hasRetransmission && retransmitHandshakePacket != nil {
			s.packer.QueueControlFrame(pth.sentPacketHandler.GetStopWaitingFrame(true), pth)
			packet, err := s.packer.PackHandshakeRetransmission(retransmitHandshakePacket, pth)
			if err != nil {
				return err
			}
			if err = s.sendPackedPacket(packet, pth); err != nil {
				return err
			}
			continue
		}

		// XXX Some automatic ACK generation should be done someway
		var ack *wire.AckFrame

		ack = pth.GetAckFrame()
		if ack != nil {
			s.packer.QueueControlFrame(ack, pth)
		}
		if ack != nil || hasStreamRetransmission {
			swf := pth.sentPacketHandler.GetStopWaitingFrame(hasStreamRetransmission)
			if swf != nil {
				s.packer.QueueControlFrame(swf, pth)
			}
		}

		// Also add CLOSE_PATH frames, if any
		for cpf := s.streamFramer.PopClosePathFrame(); cpf != nil; cpf = s.streamFramer.PopClosePathFrame() {
			s.packer.QueueControlFrame(cpf, pth)
		}

		// Also add ADD ADDRESS frames, if any
		for aaf := s.streamFramer.PopAddAddressFrame(); aaf != nil; aaf = s.streamFramer.PopAddAddressFrame() {
			s.packer.QueueControlFrame(aaf, pth)
		}

		// Also add PATHS frames, if any
		for pf := s.streamFramer.PopPathsFrame(); pf != nil; pf = s.streamFramer.PopPathsFrame() {
			s.packer.QueueControlFrame(pf, pth)
		}

		pkt, sent, err := sch.performPacketSending(s, windowUpdateFrames, pth)
		if err != nil {
			if err == ackhandler.ErrTooManyTrackedSentPackets {
				utils.Errorf("Closing episode")
				if sch.SchedulerName == "dqnAgent" && sch.Training {
					sch.TrainingAgent.CloseEpisode(uint64(s.connectionID), -100, false)
				}
			}
			return err
		}
		windowUpdateFrames = nil
		if !sent {
			// Prevent sending empty packets
			return sch.ackRemainingPaths(s, windowUpdateFrames)
		}

		// Duplicate traffic when it was sent on an unknown performing path
		// FIXME adapt for new paths coming during the connection
		if pth.rttStats.SmoothedRTT() == 0 {
			currentQuota := sch.quotas[pth.pathID]
			// Was the packet duplicated on all potential paths?
		duplicateLoop:
			for pathID, tmpPth := range s.paths {
				if pathID == protocol.InitialPathID || pathID == pth.pathID {
					continue
				}
				if sch.quotas[pathID] < currentQuota && tmpPth.sentPacketHandler.SendingAllowed() {
					// Duplicate it
					pth.sentPacketHandler.DuplicatePacket(pkt)
					break duplicateLoop
				}
			}
		}

		// And try pinging on potentially failed paths
		if fromPth != nil && fromPth.potentiallyFailed.Get() {
			err = s.sendPing(fromPth)
			if err != nil {
				return err
			}
		}
	}
}

func PrintSchedulerInfo(config *Config) {
	// Scheduler Info
	schedulerList := []string{constants.SCHEDULER_ROUND_ROBIN, constants.SCHEDULER_SCH_SHRAVAN, constants.SCHEDULER_LOW_LATENCY,
		constants.SCHEDULER_PEEKABOO, constants.SCHEDULER_ECF, constants.SCHEDULER_AGENT, constants.SCHEDULER_BLEST,
		constants.SCHEDULER_FIRST_PATH, constants.SCHEDULER_LOW_BANDIT, constants.SCHEDULER_RANDOM, constants.SCHEDULER_DEAR, constants.SCHEDULER_MAC2, constants.SCHEDULER_MAC3, constants.SCHEDULER_CAP, constants.SCHEDULER_PACE, constants.SCHEDULER_KUMAR, constants.SCHEDULER_PUMAR,constants.SCHEDULER_A2C,constants.SCHEDULER_PPO}
	if config.Scheduler == "" {
		fmt.Println("Using Default Multipath Scheduler: ", constants.SCHEDULER_ROUND_ROBIN)
	} else if util.StringInSlice(schedulerList, config.Scheduler) {
		fmt.Println("Selected Multipath Scheduler:", config.Scheduler)
	} else {
		fmt.Printf("Invalid Multipath Scheduler selected, defaulting to %s\n Available schedulers: %s\n",
			constants.SCHEDULER_ROUND_ROBIN, schedulerList)
	}
}

