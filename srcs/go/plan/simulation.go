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

func getBackupTypeFromEnv() bool {
	val, _ := os.LookupEnv(kb.ActiveBackupEnvKey)
	if val == "true" {
		return true
	}
	return false
}
