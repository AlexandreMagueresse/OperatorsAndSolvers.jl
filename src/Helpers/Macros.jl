const msg_abstractmethod = """
This function belongs to an abstract API and has not been implemented yet.
"""

"""
    @abstractmethod(message::AbstractString)

Macro used to describe an abstract method part of an API.
"""
macro abstractmethod(message::AbstractString=msg_abstractmethod)
  quote
    error($(esc(message)))
  end
end

const msg_unreachable = """
This line cannot be reached.
"""

"""
    @unreachable(message::AbstractString)

Macro used to label a line that cannot be reached.
"""
macro unreachable(message::AbstractString=msg_unreachable)
  quote
    error($(esc(message)))
  end
end
