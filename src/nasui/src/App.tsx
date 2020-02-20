import React, { ChangeEvent } from 'react';
import './App.css';
import 'typeface-roboto';
import { createStyles, withStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Grid from '@material-ui/core/Grid';
import IconButton from '@material-ui/core/IconButton';
import Slider from '@material-ui/core/Slider';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import RefreshIcon from '@material-ui/icons/Refresh';
import Chart from './Chart';

const styles = createStyles({
  bottomAppBar: {
    top: 'auto',
    bottom: 0,
  },
  title: {
    flexGrow: 1,
    textAlign: 'left'
  },
});

type AppState = {
  graphData: any,
  logData: any[],
  sliderValue: number,
  maxSliderValue: number,
}

type AppProps = {
  classes: any
}

class App extends React.Component<AppProps, AppState>  {
  constructor(props: any) {
    super(props);
    this.state = {
      graphData: null,
      logData: [],
      sliderValue: 0,
      maxSliderValue: 0
    };
    this.refresh = this.refresh.bind(this);
  }

  componentDidMount() {
    this.refresh();
  }

  refresh() {
    fetch("/refresh")
      .then((response) => { return response.json() })
      .then((data) => {
        this.setState({
          graphData: data["graph"],
          logData: data["log"],
          maxSliderValue: data["log"].length,
          sliderValue: Math.min(data["log"].length, this.state.sliderValue)
        });
      });
  }

  render() {
    const { classes } = this.props;
    const { sliderValue, maxSliderValue } = this.state;
    const handleSliderChange = (event: ChangeEvent<{}>, value: number | number[]) => {
      this.setState({ sliderValue: value as number });
    };
    return (
      <div className='App'>
        <AppBar position='static' color='primary'>
          <Toolbar>
            <Typography variant='h6' className={classes.title}>
              NNI NAS Board
            </Typography>
            <IconButton color="inherit" onClick={this.refresh}>
              <RefreshIcon />
            </IconButton>
          </Toolbar>
        </AppBar>
        <AppBar position='fixed' color='default' className={classes.bottomAppBar}>
          <Toolbar variant='dense'>
            <Grid container spacing={2} alignItems='center'>
              <Grid item xs>
                <Slider
                  value={sliderValue}
                  max={maxSliderValue}
                  min={0}
                  step={1}
                  onChange={handleSliderChange}
                />
              </Grid>
              <Grid item>
                <Typography variant='body1'>
                  {sliderValue}/{maxSliderValue}
                </Typography>
              </Grid>
            </Grid>
          </Toolbar>
        </AppBar>
        <Chart width={window.innerWidth} height={window.innerHeight} displayStep={sliderValue}
          graphData={this.state.graphData} logData={this.state.logData} />
      </div>
    );
  }
}

export default withStyles(styles)(App);
