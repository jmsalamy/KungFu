package plan

import (
	"os"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

func getDelayFromEnv() bool {
	val, _ := os.LookupEnv(kb.DelayOnEnvKey)
	if val == "true" {
		return true
	}
	return false
}
