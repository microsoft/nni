import React, { ChangeEvent } from 'react';
import './App.css';
import 'typeface-roboto';
import { createStyles, withStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';
import IconButton from '@material-ui/core/IconButton';
import Slider from '@material-ui/core/Slider';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import RefreshIcon from '@material-ui/icons/Refresh';
import SettingsIcon from '@material-ui/icons/Settings';
import FormControl from '@material-ui/core/FormControl';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormGroup from '@material-ui/core/FormGroup';
import Checkbox from '@material-ui/core/Checkbox';
import Dialog from '@material-ui/core/Dialog';
import DialogTitle from '@material-ui/core/DialogTitle';
import DialogContent from '@material-ui/core/DialogContent';
import DialogActions from '@material-ui/core/DialogActions';
import ExpansionPanel from '@material-ui/core/ExpansionPanel';
import ExpansionPanelSummary from '@material-ui/core/ExpansionPanelSummary';
import ExpansionPanelDetails from '@material-ui/core/ExpansionPanelDetails';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import Chart from './Chart';
import { preprocessGraphData, eliminateSidechainNodes, subgraphMetadata } from './graphUtils';
import lodash from 'lodash';

const styles = createStyles({
  bottomAppBar: {
    top: 'auto',
    bottom: 0,
  },
  title: {
    flexGrow: 1,
    textAlign: 'left'
  },
  panel: {
    position: 'absolute',
    top: 76,
    right: 16,
    width: 400,
  },
  listItem: {
    paddingLeft: 0,
    paddingRight: 0,
    paddingTop: 2,
    paddingBottom: 2,
    fontSize: '80%'
  },
  listSubtitle: {
    fontWeight: 600,
    paddingLeft: 0,
    paddingRight: 0
  }
});

type AppState = {
  rawGraphData: any,
  graphData: any,
  logData: any[],
  sliderValue: number,
  maxSliderValue: number,
  sliderStep: number,
  settingsOpen: boolean,
  hideSidechainNodes: boolean,
  hidePrimitiveNodes: boolean,
  selectedNode: string,
}

type AppProps = {
  classes: any
}

class App extends React.Component<AppProps, AppState>  {
  constructor(props: any) {
    super(props);
    this.state = {
      rawGraphData: null,
      graphData: null,
      logData: [],
      sliderValue: 0,
      maxSliderValue: 0,
      sliderStep: 1,
      settingsOpen: false,
      hideSidechainNodes: true,
      hidePrimitiveNodes: true,
      selectedNode: '',
    };
    this.refresh = this.refresh.bind(this);
  }

  componentDidMount() {
    this.refresh();
  }

  refresh() {
    fetch('/refresh')
      .then((response) => { return response.json() })
      .then((data) => {
        this.setState({
          rawGraphData: data['graph'],
          graphData: this.graphProcessPipeline(data['graph']),
          logData: data['log'],
          maxSliderValue: data['log'].length - 1,
          sliderStep: Math.max(1, Math.floor(data['log'].length / 20)),
          sliderValue: Math.min(data['log'].length, this.state.sliderValue)
        });
      });
  }

  private graphProcessPipeline(rawGraph: any): any {
    const graph = lodash.cloneDeep(rawGraph);
    if (this.state.hideSidechainNodes)
      eliminateSidechainNodes(graph);
    preprocessGraphData(graph);
    return graph;
  }

  private renderExpansionPanel() {
    const { classes } = this.props;
    const { selectedNode, graphData } = this.state;
    if (graphData === null)
      return null;
    const info = subgraphMetadata(graphData, selectedNode);
    if (info === null)
      return null;
    const subtitle = info.op ?
    (info.op === 'IO Node' ? info.op : `Operation: ${info.op}`) :
      `Subgraph: ${info.nodeCount} nodes, ${info.edgeCount} edges`;
    return (
      <ExpansionPanel className={classes.panel}>
        <ExpansionPanelSummary
          expandIcon={<ExpandMoreIcon />}
        >
          <Typography variant='subtitle1'>{info.name}</Typography>
        </ExpansionPanelSummary>
        <ExpansionPanelDetails>
          <List dense={true} style={{
            maxHeight: window.innerHeight * .5,
            overflowY: 'scroll',
            wordWrap: 'break-word',
          }}>
            <ListItem className={classes.listSubtitle}>{subtitle}</ListItem>
            <ListItem className={classes.listSubtitle}>Attributes</ListItem>
            <ListItem className={classes.listItem}>{info.attributes}</ListItem>
            <ListItem className={classes.listSubtitle}>Inputs</ListItem>
            {
              info.inputs.map(item => <ListItem className={classes.listItem}>{item}</ListItem>)
            }
            <ListItem className={classes.listSubtitle}>Outputs</ListItem>
            {
              info.inputs.map(item => <ListItem className={classes.listItem}>{item}</ListItem>)
            }
          </List>
        </ExpansionPanelDetails>
      </ExpansionPanel>
    );
  }

  render() {
    const { classes } = this.props;
    const { sliderValue, maxSliderValue, sliderStep, settingsOpen } = this.state;
    const handleSliderChange = (event: ChangeEvent<{}>, value: number | number[]) => {
      this.setState({ sliderValue: value as number });
    };
    const handleSettingsDialogToggle = (value: boolean) => () => {
      this.setState({ settingsOpen: value });
    };
    const handleSettingsChange = (name: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
      this.setState({
        ...this.state,
        [name]: event.target.checked
      }, () => {
        this.setState({
          graphData: this.graphProcessPipeline(this.state.rawGraphData),
        })
      });
    };
    const handleSelectionChange = (node: string) => {
      this.setState({
        selectedNode: node
      });
    };
    return (
      <div className='App'>
        <Chart 
          width={window.innerWidth}
          height={window.innerHeight}
          displayStep={sliderValue}
          graphData={this.state.graphData}
          logData={this.state.logData}
          handleSelectionChange={handleSelectionChange}
        />
        <AppBar position='fixed' color='primary'>
          <Toolbar>
            <Typography variant='h6' className={classes.title}>
              NNI NAS Board
            </Typography>
            <IconButton color='inherit' onClick={this.refresh}>
              <RefreshIcon />
            </IconButton>
            <IconButton color='inherit' onClick={handleSettingsDialogToggle(true)}>
              <SettingsIcon />
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
                  step={sliderStep}
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
        <Dialog onClose={handleSettingsDialogToggle(false)} open={settingsOpen}>
          <DialogTitle>Settings</DialogTitle>
          <DialogContent>
            <FormControl component="fieldset">
              <FormGroup>
                <FormControlLabel
                  control={<Checkbox checked={this.state.hideSidechainNodes}
                    onChange={handleSettingsChange('hideSidechainNodes')}
                    value='hideSidechainNodes' />}
                  label="Hide sidechain nodes"
                />
                { // TODO: hide primitive nodes
                /* <FormControlLabel
                  control={<Checkbox checked={this.state.hidePrimitiveNodes}
                    onChange={handleSettingsChange('hidePrimitiveNodes')}
                    value='hidePrimitiveNodes' />}
                  label="Hide primitive nodes"
                /> */}
              </FormGroup>
            </FormControl>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleSettingsDialogToggle(false)} color="primary">
              Close
            </Button>
          </DialogActions>
        </Dialog>
        {this.renderExpansionPanel()}
      </div>
    );
  }
}

export default withStyles(styles)(App);
