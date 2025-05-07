export type ResponseSources = {
  text: string;
  doc_id: string;
  start: number;
  end: number;
  similarity: number;
};

export type QueryResponse = {
  text: string;
  sources: ResponseSources[];
};

export const getPredictions = async (query: string): Promise<QueryResponse> => {
  const queryURL = new URL('http://localhost:5601/get_predictions');

  const response = await fetch(queryURL, { mode: 'cors' });
  if (!response.ok) {
    return { text: 'Error in query', sources: [] };
  }
  console.log(response);
  const queryResponse = await response.json();

  return queryResponse;
};

export const getForecastors = async (query: string): Promise<QueryResponse> => {
  const queryURL = new URL('http://localhost:5601/get_forecastors');

  const response = await fetch(queryURL, { mode: 'cors' });
  if (!response.ok) {
    return { text: 'Error in query', sources: [] };
  }
  console.log(response);
  const queryResponse = await response.json();

  return queryResponse;
};

const queryIndex = async (query: string): Promise<QueryResponse> => {
  const queryURL = new URL('http://localhost:5601/query?');
  queryURL.searchParams.append('text', query);

  const response = await fetch(queryURL, { mode: 'cors' });
  if (!response.ok) {
    return { text: 'Error in query', sources: [] };
  }
  console.log(response);

  const queryResponse = await response.json();

  return queryResponse;
};

export default queryIndex;
