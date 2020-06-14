import React, { useState, useEffect } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import FilledInput from '@material-ui/core/FilledInput';
import InputLabel from '@material-ui/core/InputLabel';
import FormControl from '@material-ui/core/FormControl';
import FormHelperText from '@material-ui/core/FormHelperText';
import NativeSelect from '@material-ui/core/NativeSelect';

const models = [
  'unigram',
  'bigram',
  'trigram',
  'trigram_backoff',
  'trigram_kn_backoff',
  'neural_trigram',
  'LSTM',
];

const useStyles = makeStyles(() => ({
  app: {
    display: 'grid',
    padding: 16,
    textAlign: 'center',
    justifyContent: 'center',
  },
  boldLetter: {
    fontWeight: 'bold',
  },
  dropdown: {
    alignItems: 'center',
    display: 'flex',
    justifyContent: 'space-around',
  },
  textField: {
    alignItems: 'center',
    display: 'flex',
    justifyContent: 'center',
    width: 1000,
  },
  nextWord: {
    color: 'red',
  },
  subtitle: {
    color: 'gray',
    fontSize: '16px',
    padding: 16,
    textAlign: 'center',
  },
  title: {
    fontSize: '32px',
    textAlign: 'center',
  },
}));

const App = () => {
  const classes = useStyles();
  const [model, setModel] = useState('None');
  const [nextWord, setNextWord] = useState('');
  const [perplexity, setPerplexity] = useState('');

  useEffect(() => {
    const url = `/get_perplexity?model_name=${model}`;

    fetch(url, { method: 'GET' }).then((response) => {
      response.json().then((data) => {
        setPerplexity(data);
      });
    });
  }, [model]);

  const onTextChange = ((event) => {
    const url = '/next_word';
    const text = event.target.value;
    if (text[text.length - 1] !== ' ') {
      return;
    }
    setNextWord('loading...');
    fetch(url, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model_name: model, text }),
    }).then((response) => {
      response.json().then((data) => {
        setNextWord(data);
      });
    });
  });

  const handleDropDownChange = (event) => {
    setModel(event.target.value);
  };

  const onKeyDown = (event) => {
    event.persist();
    if (event.keyCode === 9) {
      event.preventDefault();
      // eslint-disable-next-line no-param-reassign
      event.target.value += nextWord;
      setNextWord('');
    }
  };

  return (
    <div className={classes.app}>
      <div className={classes.title}>
        autocomp
        <span className={classes.boldLetter}>L</span>
        ete
        <span className={classes.boldLetter}> M</span>
        e
      </div>
      <div className={classes.subtitle}>Interact with Language Models</div>
      <div className={classes.dropdown}>
        <InputLabel htmlFor="age-native-helper">Pick a Language Model</InputLabel>
        <FormControl className={classes.formControl}>
          <NativeSelect
            value={model}
            onChange={handleDropDownChange}
          >
            <option key="None" value="None">None</option>
            {models.map((modelName) => <option key={modelName} value={modelName}>{modelName}</option>)}
          </NativeSelect>
          <FormHelperText>
            {`perplexity: ${perplexity}`}
          </FormHelperText>
        </FormControl>
      </div>
      <div className={classes.textField}>
        <FilledInput color="primary" onChange={onTextChange} onKeyDown={onKeyDown} fullWidth multiline />
      </div>
      <div>
        Suggested next word:
        {' '}
        {nextWord}
      </div>
    </div>
  );
};

export default App;
