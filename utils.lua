local utils = {}

--[[
  Prints the time and a message in the form of

  <time> <message>

  example: 08:58:23 Hello World!
]]--
function utils.printTime(message)
  local timeObject = os.date("*t")
  local currTime = ("%02d:%02d:%02d"):format(timeObject.hour, timeObject.min, timeObject.sec)
  print("%s %s" % {currTime, message})
end

return utils