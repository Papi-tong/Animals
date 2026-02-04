const cloud = require("wx-server-sdk")
cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV })

exports.main = async (event, context) => {
  const { fileID } = event || {}
  if (!fileID) return { success: false, errMsg: "fileID required" }
  return { success: true, label: "unknown", score: 0 }
}