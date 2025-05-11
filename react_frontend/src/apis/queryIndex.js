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

export const queryStreamingIndex = async (query: string): Promise<Response> => {

    const response = await fetch('http://127.0.0.1:8080/run-buybak-workflow', {
        method: "POST",
        body: JSON.stringify({
            query: query,
            stream: true
        }),
        headers: {
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
    }, { mode: 'no-cors' });

    console.log({response});

    return response;
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
