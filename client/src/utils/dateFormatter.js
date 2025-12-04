export const formatHour = (hour) => {
  const h = hour % 12 || 12;
  const ampm = hour < 12 ? 'AM' : 'PM';
  return `${h}:00 ${ampm}`;
};

export const formatWeekday = (day) => {
  const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  return days[day] || day;
};

export const formatPersonaName = (persona) => {
  return persona.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());
};
